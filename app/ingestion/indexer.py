"""索引构建：多格式解析 → 父子分层切块 → 向量化 → 原子发布。

为什么要有「原子发布」
----------------------
原实现是「先删旧库 → 再重建」。如果中途失败（模型下载失败、磁盘满、进程被杀），
用户就只剩一个空库，而且旧索引已经没了。这里改成两阶段：

    build 阶段：所有写入都落到 __staging 集合与 staging 父块目录
    commit 阶段：删除旧集合 → 把 staging 重命名为正式名 → 切换父块目录

这样构建失败时旧索引完好无损，成功时切换是瞬时的。

父块目录的切换用「重命名」而非「复制」，是为了避免大语料下的长时间拷贝；
重命名在同一文件系统内是原子的。
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

try:  # 包方式：app.ingestion.indexer
    from ..chunker import DocumentChunker, ParentStore
except ImportError:  # 扁平方式：app/ 已加入 sys.path，ingestion 是顶层包
    from chunker import DocumentChunker, ParentStore  # type: ignore[no-redef]

from .loaders import ParsedDocument, parse_bytes
from .manifest import (
    FileFingerprint,
    IndexManifest,
    ManifestDiff,
    build_fingerprints,
    content_hash,
    now_iso,
)

logger = logging.getLogger(__name__)

STAGING_SUFFIX = "__staging"


@dataclass
class IndexReport:
    """一次索引构建的结果，直接回给 API 与前端。"""

    ok: bool
    mode: str                      # full / incremental / noop
    child_count: int = 0
    parent_count: int = 0
    parsed_files: int = 0
    skipped_files: list[dict] = field(default_factory=list)
    failed_files: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    diff: ManifestDiff | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "child_count": self.child_count,
            "parent_count": self.parent_count,
            "parsed_files": self.parsed_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "warnings": self.warnings,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "diff": (
                {
                    "added": self.diff.added,
                    "changed": self.diff.changed,
                    "removed": self.diff.removed,
                    "unchanged_count": len(self.diff.unchanged),
                }
                if self.diff
                else None
            ),
        }


def _chunk_config(settings) -> dict:
    """影响切块结果的参数。任一变化都会使旧索引不再等价。"""
    return {
        "parent_min": settings.parent_chunk_size_min,
        "parent_max": settings.parent_chunk_size_max,
        "child_size": settings.child_chunk_size,
        "child_overlap": settings.child_chunk_overlap,
    }


def scan_corpus(settings) -> tuple[dict[str, dict], list[dict]]:
    from .loaders import SUPPORTED_SUFFIXES

    return build_fingerprints(settings.docs_dir, settings.corpus_patterns, SUPPORTED_SUFFIXES)


def current_manifest(settings) -> IndexManifest | None:
    return IndexManifest.load(settings.manifest_path)


def compute_diff(settings) -> tuple[ManifestDiff | None, list[dict]]:
    """当前语料相对已发布清单的差异。无清单时返回 None（表示需要全量构建）。"""
    manifest = current_manifest(settings)
    fingerprints, failures = scan_corpus(settings)
    if manifest is None:
        return None, failures
    return manifest.diff_against(fingerprints), failures


def parse_corpus(
    settings, only: set[str] | None = None
) -> tuple[list[ParsedDocument], list[dict], list[str]]:
    """解析语料目录下的文件。

    only 非空时只解析其中的相对路径（增量模式）。返回 (文档, 失败, 警告)。
    """
    from .loaders import SUPPORTED_SUFFIXES

    fingerprints, scan_failures = build_fingerprints(
        settings.docs_dir, settings.corpus_patterns, SUPPORTED_SUFFIXES
    )
    documents: list[ParsedDocument] = []
    failures: list[dict] = list(scan_failures)
    warnings: list[str] = []

    for rel, meta in sorted(fingerprints.items()):
        if only is not None and rel not in only:
            continue
        path = settings.docs_dir / rel
        try:
            parsed = parse_bytes(path.read_bytes(), rel)
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不中断整批
            failures.append({"rel_path": rel, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        for warning in parsed.warnings:
            warnings.append(f"{rel}：{warning}")
        if not parsed.text.strip():
            failures.append({"rel_path": rel, "reason": "解析后正文为空"})
            continue
        documents.append(parsed)

    return documents, failures, warnings


def _chunk_documents(settings, documents: list[ParsedDocument]):
    """把解析结果切成父块与子块，并把页码写进元数据。"""
    from .loaders import page_of

    chunker = DocumentChunker(
        min_parent_size=settings.parent_chunk_size_min,
        max_parent_size=settings.parent_chunk_size_max,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )
    parents, children = [], []
    for document in documents:
        doc_parents, doc_children = chunker.split_document(
            document.text, document.file_name, document.rel_path
        )
        for _, parent in doc_parents:
            parent.metadata.setdefault("file_type", document.file_type)
            page = page_of(parent.page_content)
            if page:
                parent.metadata["page"] = page
        for child in doc_children:
            child.metadata.setdefault("file_type", document.file_type)
            page = page_of(child.page_content)
            if page:
                child.metadata["page"] = page
        parents.extend(doc_parents)
        children.extend(doc_children)
    return parents, children


def build_index(settings, force: bool = False, incremental: bool = True) -> IndexReport:
    """构建索引。默认做增量判断；force=True 时无条件全量重建。"""
    import time

    from langchain_chroma import Chroma

    try:
        from ..embeddings import get_embeddings
    except ImportError:
        from embeddings import get_embeddings  # type: ignore[no-redef]

    started = time.time()
    report = IndexReport(ok=False, mode="full")

    fingerprints, scan_failures = scan_corpus(settings)
    report.failed_files.extend(scan_failures)

    if not fingerprints:
        report.warnings.append(
            f"语料目录为空：{settings.docs_dir}（支持 .md/.txt/.csv/.pdf/.docx）"
        )
        report.elapsed_seconds = time.time() - started
        return report

    manifest = current_manifest(settings)
    chunk_config = _chunk_config(settings)

    # ── 增量判断 ──────────────────────────────────────────────
    if manifest is not None and not force:
        compatible, reason = manifest.is_compatible(settings.embed_model, chunk_config)
        if not compatible:
            report.warnings.append(f"需全量重建：{reason}")
        else:
            diff = manifest.diff_against(fingerprints)
            report.diff = diff
            if not diff.needs_rebuild:
                report.ok = True
                report.mode = "noop"
                report.child_count = manifest.child_count
                report.parent_count = manifest.parent_count
                report.parsed_files = len(manifest.files)
                report.elapsed_seconds = time.time() - started
                return report
            report.mode = "incremental" if incremental else "full"

    # ── 增量模式：当前先退化为全量重建 ─────────────────────────
    # 说明：真正的「只重算变化文件」需要按 rel_path 从 Chroma 删除旧向量。
    # 该能力依赖稳定的 chunk id 方案，属于下一步工作；此处先保证正确性：
    # 一旦检测到变化就重建，并如实告知调用方实际执行的是 full。
    if report.mode == "incremental":
        report.warnings.append("增量写入尚未启用，本次执行全量重建（保证一致性）")
        report.mode = "full"

    staging_collection = settings.collection_name + STAGING_SUFFIX
    staging_parent_dir = settings.parent_store_dir.parent / (
        settings.parent_store_dir.name + STAGING_SUFFIX
    )

    # ── build 阶段：全部写入 staging ──────────────────────────
    documents, failures, warnings = parse_corpus(settings)
    report.failed_files.extend(failures)
    report.warnings.extend(warnings)
    report.parsed_files = len(documents)

    if not documents:
        report.warnings.append("没有解析出任何可用文档，已保留原有索引不动")
        report.elapsed_seconds = time.time() - started
        return report

    parents, children = _chunk_documents(settings, documents)
    if not children:
        report.warnings.append("切块后没有任何子块，已保留原有索引不动")
        report.elapsed_seconds = time.time() - started
        return report

    if staging_parent_dir.exists():
        shutil.rmtree(staging_parent_dir, ignore_errors=True)
    staging_store = ParentStore(staging_parent_dir)
    staging_store.save_many(parents)

    embeddings = get_embeddings()
    try:
        client = Chroma(
            collection_name=staging_collection,
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
        )
        try:
            client.delete_collection()
        except Exception:  # noqa: BLE001 —— 首次构建时集合不存在
            pass
        client = Chroma(
            collection_name=staging_collection,
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
        )
        batch = 200
        for i in range(0, len(children), batch):
            client.add_documents(children[i : i + batch])
            logger.info("写入 staging：%d / %d", min(i + batch, len(children)), len(children))
    except Exception as exc:  # noqa: BLE001 —— build 失败必须保留旧索引
        report.warnings.append(f"staging 写入失败，旧索引保持不变：{type(exc).__name__}: {exc}")
        shutil.rmtree(staging_parent_dir, ignore_errors=True)
        report.elapsed_seconds = time.time() - started
        return report

    # ── commit 阶段：切换正式索引 ─────────────────────────────
    try:
        _commit(settings, staging_collection, staging_parent_dir, embeddings)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"索引切换失败：{type(exc).__name__}: {exc}")
        report.elapsed_seconds = time.time() - started
        return report

    new_manifest = IndexManifest(
        built_at=now_iso(),
        collection_name=settings.collection_name,
        embed_model=settings.embed_model,
        rerank_model=settings.rerank_model if settings.use_rerank else "",
        chunk_config=chunk_config,
        files=[FileFingerprint(**meta).to_dict() for meta in fingerprints.values()],
        child_count=len(children),
        parent_count=len(parents),
        failed_files=report.failed_files,
    )
    new_manifest.save(settings.manifest_path)

    report.ok = True
    report.child_count = len(children)
    report.parent_count = len(parents)
    report.elapsed_seconds = time.time() - started
    logger.info(
        "索引已发布：父块 %d / 子块 %d / 文件 %d", len(parents), len(children), len(documents)
    )
    return report


def _commit(settings, staging_collection: str, staging_parent_dir: Path, embeddings) -> None:
    """把 staging 索引切换为正式索引。"""
    from langchain_chroma import Chroma

    client = Chroma(
        collection_name=staging_collection,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )

    # 1) 删除正式集合（若存在），再把 staging 重命名为正式名
    try:
        live = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
        )
        live.delete_collection()
    except Exception:  # noqa: BLE001 —— 首次构建时正式集合不存在
        pass
    client._collection.modify(name=settings.collection_name)

    # 2) 父块目录切换：旧目录改名保留一份，新目录就位
    backup = settings.parent_store_dir.parent / (settings.parent_store_dir.name + "__old")
    shutil.rmtree(backup, ignore_errors=True)
    if settings.parent_store_dir.exists():
        settings.parent_store_dir.rename(backup)
    staging_parent_dir.rename(settings.parent_store_dir)
    shutil.rmtree(backup, ignore_errors=True)


def delete_document(settings, rel_path: str) -> IndexReport:
    """删除一个源文件并重建索引。

    先删磁盘文件，再重建；重建失败时文件已删除但索引仍是旧的 —— 这种不一致会
    被下一次构建的 diff 检出，因此是安全方向（宁可少一份文档，也不要幽灵索引）。
    """
    target = settings.docs_dir / rel_path
    if not target.exists():
        return IndexReport(ok=False, mode="delete", warnings=[f"文件不存在：{rel_path}"])
    target.unlink()
    return build_index(settings, force=False, incremental=True)


def save_upload(settings, rel_path: str, raw: bytes) -> tuple[bool, str]:
    """保存上传文件到语料目录。返回 (是否成功, 消息)。

    会校验类型与体积，并拒绝路径穿越。
    """
    from .loaders import MAX_UPLOAD_BYTES, SUPPORTED_SUFFIXES, detect_file_type

    normalized = Path(rel_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return False, "非法的文件路径"
    if normalized.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False, f"不支持的文件类型：{normalized.suffix}"
    try:
        detect_file_type(normalized.name)
    except ValueError as exc:
        return False, str(exc)
    if len(raw) > MAX_UPLOAD_BYTES:
        return False, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限"
    if not raw:
        return False, "文件内容为空"

    target = settings.docs_dir / normalized
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return True, f"已保存 {normalized.as_posix()}（{len(raw)} 字节，指纹 {content_hash(raw)}）"
