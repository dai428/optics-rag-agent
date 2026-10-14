"""文档接入层：多格式解析、增量索引与原子发布。"""

from .indexer import (
    IndexReport,
    build_index,
    compute_diff,
    current_manifest,
    delete_document,
    parse_corpus,
    save_upload,
    scan_corpus,
)
from .loaders import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_SUFFIXES,
    DocumentTooLargeError,
    ParsedDocument,
    UnsupportedFormatError,
    detect_file_type,
    page_of,
    parse_bytes,
    parse_path,
    strip_page_markers,
)
from .manifest import (
    FileFingerprint,
    IndexManifest,
    ManifestDiff,
    build_fingerprints,
    content_hash,
)

__all__ = [
    "IndexReport",
    "build_index",
    "compute_diff",
    "current_manifest",
    "delete_document",
    "parse_corpus",
    "save_upload",
    "scan_corpus",
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_SUFFIXES",
    "DocumentTooLargeError",
    "ParsedDocument",
    "UnsupportedFormatError",
    "detect_file_type",
    "page_of",
    "parse_bytes",
    "parse_path",
    "strip_page_markers",
    "FileFingerprint",
    "IndexManifest",
    "ManifestDiff",
    "build_fingerprints",
    "content_hash",
]
