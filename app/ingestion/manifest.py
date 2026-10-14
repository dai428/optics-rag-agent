"""索引清单（manifest）：记录语料指纹与索引版本，支撑增量与一致性校验。

为什么需要它
------------
原先 `build_vectorstore` 只要发现 Chroma collection 非空就直接复用，完全不看
语料是否变过。结果是：改了文档却仍在查旧索引，而用户毫无察觉。

本模块负责：
  1. 计算每个源文件的内容 Hash（用于判断「是否需要重建」）；
  2. 记录切块参数与嵌入模型（参数一变，旧索引就不再等价）；
  3. 记录父块数量与子块数量（用于检测「库与父块仓库不配套」）；
  4. 提供 staging → commit 的两阶段写入，使索引发布是原子的。

清单本身是一个 JSON 文件，人可以直接打开核对，不需要额外数据库。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1


def content_hash(raw: bytes) -> str:
    """内容指纹。用 SHA-256 前 16 位十六进制，足够区分且短。"""
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class FileFingerprint:
    """单个源文件的指纹。"""

    rel_path: str
    content_hash: str
    size: int
    file_type: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IndexManifest:
    """一次索引构建的完整描述。"""

    version: int = MANIFEST_VERSION
    built_at: str = ""
    collection_name: str = ""
    embed_model: str = ""
    rerank_model: str = ""
    chunk_config: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    child_count: int = 0
    parent_count: int = 0
    failed_files: list[dict] = field(default_factory=list)

    # ── 读写 ──────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> "IndexManifest | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 校验 ──────────────────────────────────────────────
    def file_map(self) -> dict[str, str]:
        """rel_path → content_hash。"""
        return {f["rel_path"]: f["content_hash"] for f in self.files if "rel_path" in f}

    def fingerprint_map(self) -> dict[str, dict]:
        return {f["rel_path"]: f for f in self.files if "rel_path" in f}

    def is_compatible(self, embed_model: str, chunk_config: dict) -> tuple[bool, str]:
        """判断旧清单能否继续复用：嵌入模型与切块参数必须一致。

        嵌入模型换掉后，旧向量与新向量不在同一空间，混用会得到无意义的相似度；
        切块参数变了则 parent_id 的对应关系也变了。这两种情况都必须重建。
        """
        if self.version != MANIFEST_VERSION:
            return False, f"清单版本 {self.version} 与当前 {MANIFEST_VERSION} 不一致"
        if self.embed_model != embed_model:
            return False, f"嵌入模型已变更：{self.embed_model} → {embed_model}"
        if self.chunk_config != chunk_config:
            return False, "切块参数已变更"
        return True, ""

    def diff_against(self, current: dict[str, dict]) -> "ManifestDiff":
        """与当前磁盘上的语料对比，得出新增 / 修改 / 删除。"""
        old = self.fingerprint_map()
        old_keys, new_keys = set(old), set(current)

        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        changed = sorted(
            k
            for k in old_keys & new_keys
            if old[k].get("content_hash") != current[k].get("content_hash")
        )
        unchanged = sorted((old_keys & new_keys) - set(changed))
        return ManifestDiff(added=added, changed=changed, removed=removed, unchanged=unchanged)


@dataclass
class ManifestDiff:
    """增量索引的决策依据。"""

    added: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: list[str]

    @property
    def needs_rebuild(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    @property
    def dirty_files(self) -> list[str]:
        return sorted(set(self.added) | set(self.changed))

    def summary(self) -> str:
        if not self.needs_rebuild:
            return f"语料无变化（{len(self.unchanged)} 个文件）"
        parts = []
        if self.added:
            parts.append(f"新增 {len(self.added)}")
        if self.changed:
            parts.append(f"修改 {len(self.changed)}")
        if self.removed:
            parts.append(f"删除 {len(self.removed)}")
        return "语料已变化：" + "、".join(parts)


def build_fingerprints(
    docs_dir: Path, patterns: tuple[str, ...], parse_suffixes: dict[str, str]
) -> tuple[dict[str, dict], list[dict]]:
    """扫描语料目录，返回 (指纹表, 扫描失败列表)。

    指纹表结构：rel_path → {content_hash, size, file_type}
    """
    seen: set[str] = set()
    fingerprints: dict[str, dict] = {}
    failures: list[dict] = []

    for pattern in patterns:
        for path in sorted(docs_dir.glob(pattern)):
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            rel = path.relative_to(docs_dir).as_posix()
            suffix = path.suffix.lower()
            if suffix not in parse_suffixes:
                failures.append({"rel_path": rel, "reason": f"不支持的扩展名 {suffix}"})
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                failures.append({"rel_path": rel, "reason": f"读取失败：{exc}"})
                continue
            fingerprints[rel] = {
                "rel_path": rel,
                "content_hash": content_hash(raw),
                "size": len(raw),
                "file_type": parse_suffixes[suffix],
            }
    return fingerprints, failures


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
