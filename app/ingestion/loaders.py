"""多格式文档解析：把不同来源统一成「可切块的文本 + 结构化元数据」。

设计要点
--------
1. **统一为 Markdown 化文本**
   下游 chunker 依赖 Markdown 标题做父子分层。因此 DOCX 的 Heading 样式会被
   转成 `#` / `##`，CSV 会被转成 Markdown 表格（这样表格展开逻辑才能生效）。

2. **保留可定位信息**
   PDF 没有标题结构，但「第几页」是科研材料最重要的定位锚点。做法是在页与页
   之间插入 `[第 N 页]` 标记行：它既是人类可读的正文，也能被下游解析回 page
   字段写进元数据，从而支持「引用到页码」。

3. **失败要可见，不要静默**
   每个解析器返回 `warnings`，由调用方决定是否把该文件标记为「部分成功」。
   这比 `except: pass` 之后得到一份空文档要好得多。
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

PAGE_MARKER_RE = re.compile(r"^\[第\s*(\d+)\s*页\]$", re.MULTILINE)

# 支持的类型 → 解析器名。新增格式只需在这里登记并实现一个函数。
SUPPORTED_SUFFIXES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".csv": "csv",
    ".pdf": "pdf",
    ".docx": "docx",
}

# 上传体积上限（字节）。科研 PDF 通常几 MB，20 MB 足够且能挡住误传的大文件。
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass
class ParsedDocument:
    """解析结果：一份可直接交给 chunker 的规范化文档。"""

    rel_path: str
    file_name: str
    file_type: str
    text: str
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


class UnsupportedFormatError(ValueError):
    """文件类型不在支持列表内。"""


class DocumentTooLargeError(ValueError):
    """文件超过 MAX_UPLOAD_BYTES。"""


def detect_file_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise UnsupportedFormatError(f"不支持的文件类型 {suffix or '(无扩展名)'}；支持：{supported}")
    return SUPPORTED_SUFFIXES[suffix]


def _decode_text(raw: bytes) -> tuple[str, list[str]]:
    """按 UTF-8 → GB18030 → latin-1 顺序尝试解码，并记录实际使用的编码。

    中文实验材料里 GBK/GB18030 编码很常见，直接用 utf-8 读会抛异常或产生乱码。
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), []
        except UnicodeDecodeError:
            continue
    text = raw.decode("utf-8", errors="replace")
    return text, ["未能确定文本编码，已用 UTF-8 容错解码，可能含替换字符"]


def _parse_markdown(raw: bytes) -> tuple[str, int, list[str]]:
    text, warnings = _decode_text(raw)
    return text, 0, warnings


def _parse_text(raw: bytes) -> tuple[str, int, list[str]]:
    text, warnings = _decode_text(raw)
    return text, 0, warnings


def _parse_csv(raw: bytes) -> tuple[str, int, list[str]]:
    """把 CSV 转成 Markdown 表格，让下游的表格展开逻辑可以直接复用。

    原先 CSV 被当作纯文本读入，表头语义在切块时丢失，检索不到「列名 + 值」
    这种最常见的提问方式。转成 Markdown 表格后，每一行都会展开成
    「表头：值」的自包含句子。
    """
    text, warnings = _decode_text(raw)
    if not text.strip():
        return "", 0, warnings + ["CSV 文件为空"]

    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return "", 0, warnings + ["CSV 未解析出任何数据行"]

    width = max(len(r) for r in rows)
    header = [c.strip() for c in rows[0]] + [""] * (width - len(rows[0]))
    # 表头为空列时补一个占位名，否则展开后会出现无意义的「：值」
    header = [h or f"列{i + 1}" for i, h in enumerate(header)]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        padded = [c.strip() for c in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    if len(rows) == 1:
        warnings.append("CSV 只有表头、没有数据行")
    return "\n".join(lines), 0, warnings


def _parse_pdf(raw: bytes) -> tuple[str, int, list[str]]:
    """逐页抽取文本，并在页间插入 `[第 N 页]` 定位标记。"""
    from pypdf import PdfReader

    warnings: list[str] = []
    reader = PdfReader(io.BytesIO(raw))
    pages = len(reader.pages)

    chunks: list[str] = []
    empty_pages = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 —— 单页失败不应让整篇失败
            page_text = ""
            warnings.append(f"第 {index} 页文本抽取失败：{type(exc).__name__}")
        page_text = page_text.strip()
        if not page_text:
            empty_pages += 1
        chunks.append(f"[第 {index} 页]\n{page_text}")

    if empty_pages:
        warnings.append(
            f"{empty_pages}/{pages} 页未抽取到文本（可能是扫描件，需要 OCR）"
        )
    if empty_pages == pages:
        warnings.append("整篇未抽取到文本：该 PDF 很可能是扫描件，需接入 OCR 后再入库")

    return "\n\n".join(chunks), pages, warnings


def _parse_docx(raw: bytes) -> tuple[str, int, list[str]]:
    """把 Word 标题样式转成 Markdown 标题，使父子分层与章节路径继续可用。"""
    import docx

    warnings: list[str] = []
    document = docx.Document(io.BytesIO(raw))
    lines: list[str] = []
    heading_count = 0
    table_count = 0

    # 正文与表格在 XML 里是交错排列的，按文档顺序遍历才能保持阅读顺序
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            paragraph = docx.text.paragraph.Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style is not None else ""
            match = re.match(r"Heading\s*(\d+)", style, re.IGNORECASE)
            if match:
                level = min(int(match.group(1)), 3)
                lines.append("#" * level + " " + text)
                heading_count += 1
            else:
                lines.append(text)
        elif tag == "tbl":
            table = docx.table.Table(child, document)
            rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            width = max(len(r) for r in rows)
            header = [c or f"列{i + 1}" for i, c in enumerate(rows[0])]
            header += [f"列{i + 1}" for i in range(len(header), width)]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                padded = list(row) + [""] * (width - len(row))
                lines.append("| " + " | ".join(padded) + " |")
            table_count += 1
            lines.append("")

    if heading_count == 0:
        warnings.append("未识别到 Word 标题样式，章节路径将为空（建议正文使用「标题 1/2/3」样式）")
    if table_count == 0:
        warnings.append("未识别到 Word 表格")

    return "\n\n".join(lines), 0, warnings


_PARSERS = {
    "markdown": _parse_markdown,
    "text": _parse_text,
    "csv": _parse_csv,
    "pdf": _parse_pdf,
    "docx": _parse_docx,
}


def parse_bytes(raw: bytes, rel_path: str) -> ParsedDocument:
    """解析字节流。rel_path 决定类型，同时写进元数据供引用溯源。"""
    file_name = Path(rel_path).name
    file_type = detect_file_type(file_name)

    if len(raw) > MAX_UPLOAD_BYTES:
        raise DocumentTooLargeError(
            f"文件 {file_name} 为 {len(raw) / 1024 / 1024:.1f} MB，超过上限 "
            f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB"
        )
    if not raw:
        raise ValueError(f"文件 {file_name} 为空")

    parser = _PARSERS[file_type]
    text, page_count, warnings = parser(raw)

    if not text.strip():
        warnings = warnings + ["解析后正文为空，该文件不会产生任何可检索片段"]

    return ParsedDocument(
        rel_path=rel_path,
        file_name=file_name,
        file_type=file_type,
        text=text,
        page_count=page_count,
        warnings=warnings,
    )


def parse_path(path: Path, rel_path: str) -> ParsedDocument:
    return parse_bytes(path.read_bytes(), rel_path)


def page_of(text: str) -> int | None:
    """从片段内容里取回 `[第 N 页]` 标记对应的页码（取首个标记）。"""
    match = PAGE_MARKER_RE.search(text)
    return int(match.group(1)) if match else None


def strip_page_markers(text: str) -> str:
    """去掉定位标记，用于展示给用户的引用原文。"""
    return PAGE_MARKER_RE.sub("", text).strip()
