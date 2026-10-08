# D:\Desktop\rag-demo\app\chunker.py
"""父子分层切块（Parent/Child Chunking）。

解决的问题
----------
旧版用「300 字滑窗」一刀切，切出来的块有两个致命伤：
  1. Markdown 表格被拦腰截断 —— 表格行孤零零飘着，没有表头语义。
     例：Q11 问「评估网格总共多少点」，答案是 36 × 11 = 396，但 "396" 所在
     的那一行被切出来时，"波长点 / 温度点" 这些表头信息全丢了 →
     向量检索算不出相关性，必然漏。
  2. 一个语义单元（小节）被拆成多个碎片 —— 回答时只拿到半句话。

本模块的做法（分层索引）
----------------------
  ┌─ 父块（2000~4000 字）：按 Markdown 标题（# / ## / ###）切，语义完整
  │    └─ 子块（500 字 / 重叠 100）：把父块再切小，用于「精准命中」
  └─ 检索时：用【子块】算相似度（粒度小 → 向量表达聚焦 → 命中准）
             命中后回溯到【父块】拿完整上下文（语义完整 → 回答不残缺）

为什么父块不参与向量检索？
  父块太大，向量会被「稀释」——一段 3000 字的文字里只有 20 字是答案，
  它的向量和问题的相似度反而低于那 20 字单独成块的情况。
  所以父块只用来「回溯」，不做相似度计算，存 JSON 按 id 查即可。

为什么父块要合并/拆分（merge_small / split_large）？
  Markdown 标题切出来的块大小极不均匀：有的小节只有两行（<300 字），
  有的一大段 8000 字。太小则一个块信息量不足，太大则超出嵌入模型窗口。
  → 所以定一个目标区间 [min_parent_size, max_parent_size]：
     太小的往后合并，太大的再切一刀。

参考：
  GiovanniPasq/agentic-rag-for-dummies · project/document_chunker.py
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 按 Markdown 标题层级切分。key = 标题前缀，value = 写进 metadata 的字段名。
# 只切 1~3 级：`##` 是这套语料的主要分节线（研究本体那篇有 40 个 H2），
# `####` 及以下太碎，切出来会得到大量一两行的碎片块。
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def expand_markdown_tables(text: str) -> str:
    """把 Markdown 表格行展开成「自包含的自然语言句子」。

    为什么必须做这一步？
      你的语料里关键数字几乎全在 Markdown 表格里，形如：

          | 设定 | 内容 |
          |---|---|
          | 随机性 | DE 3 种子 / MC 固定 noise_seed = 999、n_mc = 2000 |

      表格行被切出来后是 `| 随机性 | DE 3 种子 ... |` —— 它**丢失了表头语义**。
      用户问「随机种子是多少」时：
        · 向量检索：表格片段语义弱，排名跌出前 40（实测）
        · BM25：问句是「随机种子」，块里是「随机性…种子」，只命中 2 个 token，被摊薄
      → 结果：含答案的块根本进不了候选集，重排再准也无用。

    展开后变成：
          随机性：DE 3 种子 / MC 固定 noise_seed = 999、n_mc = 2000
    即「表头：单元格值」。这样一句话自带完整语义，向量能懂、BM25 能整词匹配。

    实现要点
    --------
      · 以 `|` 开头的连续行视为一个表格
      · 第 1 行是表头，第 2 行是 `---|---` 分隔线（丢弃），之后是数据行
      · 每行数据按列拼成「表头：值 ｜ 表头：值」的形式
      · 非表格行原样保留（表格与正文可以混排）
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            out.append(line)
            i += 1
            continue

        # 收集连续的表格行
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1

        def _cells(row: str) -> list[str]:
            return [c.strip() for c in row.strip().strip("|").split("|")]

        if len(block) < 2:
            out.extend(block)
            continue

        headers = _cells(block[0])
        # 第 2 行通常是 |---|---| 分隔线：判断方式是所有单元格只由 -、: 组成
        data_start = 1
        second = _cells(block[1])
        if second and all(set(c) <= set("-: ") and c for c in second):
            data_start = 2

        for row in block[data_start:]:
            cells = _cells(row)
            if not any(cells):
                continue
            parts = []
            for idx, cell in enumerate(cells):
                if not cell:
                    continue
                head = headers[idx] if idx < len(headers) else ""
                # 表头为空（如空的第一列）时，只留值本身
                parts.append(f"{head}：{cell}" if head else cell)
            if parts:
                out.append(" ｜ ".join(parts))
                # ⚠️ 双保险：同时保留「竖向拼接」形式。
                #    上一行的展开是「横向」的（表头：值 ｜ 表头：值），
                #    但当表格是「键值对」型（第一列是键、第二列是值）时，
                #    竖向拼接 `键：值` 更贴近自然语言提问方式。
                #    例：`| 随机性 | MC 固定 noise_seed = 999 |`
                #        横：设定：随机性 ｜ 内容：MC 固定 noise_seed = 999
                #        竖：随机性：MC 固定 noise_seed = 999   ← 这个更利于检索
                if len(cells) == 2 and cells[0] and cells[1]:
                    out.append(f"{cells[0]}：{cells[1]}")
    return "\n".join(out)


class DocumentChunker:
    """把文档切成「父块 + 子块」两层。

    参数
    ----
    min_parent_size : 父块目标下限（字）。小于它的块尝试与邻居合并。
    max_parent_size : 父块硬上限（字）。超过它的块会被再切一刀。
    child_chunk_size : 子块大小（字）。这是真正参与向量检索的粒度。
    child_chunk_overlap : 子块重叠（字）。防止答案正好被切在边界上。
    """

    def __init__(
        self,
        min_parent_size: int = 2000,
        max_parent_size: int = 4000,
        child_chunk_size: int = 500,
        child_chunk_overlap: int = 100,
    ):
        # ── 参数自检：不合法立刻报错，别等到建库跑一半才崩 ──
        if min_parent_size <= 0 or max_parent_size < min_parent_size:
            raise ValueError("父块尺寸需为正数，且 min_parent_size <= max_parent_size")
        if not 0 <= child_chunk_overlap < child_chunk_size:
            raise ValueError("child_chunk_overlap 必须小于 child_chunk_size")
        if child_chunk_overlap >= max_parent_size:
            raise ValueError("child_chunk_overlap 必须小于 max_parent_size")

        # 延迟导入：切块器很轻，但保持与本项目其他模块一致的习惯
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,  # 保留标题文字 —— 标题是关键语义（「噪声模型」四个字本身就是线索）
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=len,
        )
        self.min_parent_size = min_parent_size
        self.max_parent_size = max_parent_size

    # ── 父块尺寸规整 ────────────────────────────────────────────────

    @staticmethod
    def _merge_metadata(target: dict, source: dict, prepend: bool = False) -> None:
        """把 source 的元数据合并进 target。

        同名键（如 h2）做字符串拼接而不是覆盖 —— 合并两个父块时，
        它们的标题路径要串起来（"台阶一 -> 台阶二"），否则丢失来源信息。
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
                continue
            first, second = (value, target[key]) if prepend else (target[key], value)
            parts = [p.strip() for raw in (first, second) for p in str(raw).split(" -> ") if p.strip()]
            target[key] = " -> ".join(dict.fromkeys(parts))  # dict.fromkeys 去重且保序

    def _merge_small_parents(self, chunks: list) -> list:
        """把尺寸不足的小父块与后续块合并，直到达到下限。"""
        if not chunks:
            return []
        merged, current = [], None
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self._merge_metadata(current.metadata, chunk.metadata)
            if len(current.page_content) >= self.min_parent_size:
                merged.append(current)
                current = None
        # 尾部余料：并进最后一个块，别单独留个孤儿
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self._merge_metadata(merged[-1].metadata, current.metadata)
            else:
                merged.append(current)
        return merged

    def _split_large_parents(self, chunks: list) -> list:
        """把超过上限的大父块再切一刀（用递归字符切分器，尽量在段落边界切）。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        result = []
        for chunk in chunks:
            if len(chunk.page_content) <= self.max_parent_size:
                result.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_parent_size,
                    chunk_overlap=0,  # 父块之间不需要重叠（重叠是给检索粒度用的）
                    length_function=len,
                )
                result.extend(splitter.split_documents([chunk]))
        return result

    def _rebalance_pair(self, first, second):
        """两个块都很小、且合并后仍不超上限时，在中间重新平衡切一刀。

        场景：标题切出来两个 1500 字的块，各自都不足 2000，但加起来 3000
        不超过 4000 —— 与其留两个「半饱」的块，不如变成一个 3000 字的完整块。
        """
        combined = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()
        lower = max(1, len(combined) - self.max_parent_size)
        upper = min(self.max_parent_size, len(combined) - 1)
        if len(combined) >= 2 * self.min_parent_size:
            lower = max(lower, self.min_parent_size)
            upper = min(upper, len(combined) - self.min_parent_size)
        if lower >= upper:
            return first, second

        preferred = min(max(len(combined) // 2, lower), upper)
        split_at = preferred
        # 优先在自然边界（空行 → 换行 → 空格）切，读起来更自然
        for sep in ("\n\n", "\n", " "):
            before = combined.rfind(sep, lower, preferred + 1)
            after = combined.find(sep, preferred, upper + 1)
            if before >= lower:
                split_at = before
                break
            if after != -1:
                split_at = after
                break

        left, right = combined[:split_at].rstrip(), combined[split_at:].lstrip()
        if not left or not right:
            return first, second

        meta = dict(first.metadata)
        self._merge_metadata(meta, second.metadata)
        first.page_content, first.metadata = left, dict(meta)
        second.page_content, second.metadata = right, dict(meta)
        return first, second

    def _clean_small_chunks(self, chunks: list) -> list:
        """收尾：仍不足下限的块，尽量塞进前一个或后一个块。"""
        cleaned = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.min_parent_size:
                if cleaned and len(cleaned[-1].page_content) + 2 + len(chunk.page_content) <= self.max_parent_size:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self._merge_metadata(cleaned[-1].metadata, chunk.metadata)
                elif i < len(chunks) - 1 and len(chunk.page_content) + 2 + len(chunks[i + 1].page_content) <= self.max_parent_size:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    self._merge_metadata(chunks[i + 1].metadata, chunk.metadata, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        # 第二轮：对仍然过小的相邻块做重平衡
        for i, chunk in enumerate(cleaned):
            if len(chunk.page_content) >= self.min_parent_size or len(cleaned) == 1:
                continue
            if i < len(cleaned) - 1:
                cleaned[i], cleaned[i + 1] = self._rebalance_pair(chunk, cleaned[i + 1])
            else:
                cleaned[i - 1], cleaned[i] = self._rebalance_pair(cleaned[i - 1], chunk)
        return cleaned

    # ── 对外主入口 ──────────────────────────────────────────────────

    def split_document(self, text: str, file_name: str, rel_path: str):
        """把一篇文档切成 (父块列表, 子块列表)。

        参数
        ----
        text : 文档全文
        file_name : 文件名（如「仿真实验总纲_2026-09-21.md」）—— 写进元数据，用于引用溯源
        rel_path  : 相对 docs_dir 的路径（如「实验报告/仿真实验总纲_2026-09-21.md」）

        返回
        ----
        parents : [(parent_id, Document), ...]
        children: [Document, ...]  —— 每个 child 的 metadata 含 parent_id 指回父块
        """
        # 先把 Markdown 表格展开成自包含句子（见 expand_markdown_tables 的说明）。
        # 注意：只在「切块前」做这一步，不改动磁盘上的原文件。
        text = expand_markdown_tables(text)

        raw_parents = self._parent_splitter.split_text(text)
        parents = self._clean_small_chunks(
            self._split_large_parents(self._merge_small_parents(raw_parents))
        )

        # 父块 id = 文件名主干 + 序号。用主干而非全路径，避免路径里的
        # 斜杠/中文在 JSON 文件名里出问题；序号保证同文件内唯一。
        stem = Path(file_name).stem
        parent_pairs, children = [], []
        for i, p in enumerate(parents):
            parent_id = f"{stem}_p{i}"
            section_path = self._section_path(p.metadata)
            meta = {
                "source": rel_path,            # 相对路径（旧字段名保留，兼容既有代码）
                "file_name": file_name,        # 文件名
                "section_path": section_path,  # 标题路径，如「第二章 理论建模 > 2.3 目标指标定义」
                "parent_id": parent_id,
                "chunk_type": "parent",
            }
            p.metadata.update(meta)
            parent_pairs.append((parent_id, p))

            # 切子块：每个子块继承父块的元数据，但 chunk_type 改为 child
            for c in self._child_splitter.split_documents([p]):
                c.metadata.update(
                    {
                        "source": rel_path,
                        "file_name": file_name,
                        "section_path": section_path,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                    }
                )
                children.append(c)

        return parent_pairs, children

    @staticmethod
    def _section_path(metadata: dict) -> str:
        """从标题元数据拼出人类可读的章节路径。

        MarkdownHeaderTextSplitter 会把标题写进 metadata 的 h1/h2/h3 字段。
        拼成 "h1 > h2 > h3" 后，回答时就能标注「出自《xx》第 2 章 2.3 节」。
        """
        parts = [str(metadata[k]).strip() for k in ("h1", "h2", "h3") if metadata.get(k)]
        return " > ".join(parts)


class ParentStore:
    """父块仓库：把父块以 JSON 落盘，按 parent_id 快速回查。

    为什么不把父块也塞进 Chroma？
      父块不参与向量检索（见模块开头的说明），塞进去只会白白增加索引体积
      和建库时间。用「一个 JSON 一个父块」的朴素方式，查询是 O(1) 文件读，
      而且人可以直接打开看，调试方便。
    """

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_many(self, parent_pairs) -> None:
        for parent_id, doc in parent_pairs:
            path = self.store_dir / f"{parent_id}.json"
            path.write_text(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def load(self, parent_id: str) -> dict | None:
        """读一个父块。不存在返回 None（不抛异常，避免检索链路因单个缺失而崩）。"""
        path = self.store_dir / f"{parent_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:  # noqa: BLE001
            logger.warning("父块读取失败 %s：%s", parent_id, e)
            return None

    def load_many(self, parent_ids: list[str]) -> list[dict]:
        """批量读父块，按传入顺序返回（去重、保持稳定排序）。"""
        seen, out = set(), []
        for pid in parent_ids:
            if pid in seen:
                continue
            seen.add(pid)
            data = self.load(pid)
            if data:
                out.append({"parent_id": pid, **data})
        return out

    def clear(self) -> None:
        """清空仓库（重建索引前调用）。"""
        for f in self.store_dir.glob("*.json"):
            f.unlink()

    def count(self) -> int:
        return len(list(self.store_dir.glob("*.json")))
