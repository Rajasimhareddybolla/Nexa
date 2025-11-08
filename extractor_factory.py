# extractor_factory.py
import os
from extractors import (
    PDFExtractor, DocxExtractor, CSVExtractor, JSONExtractor,
    TXTExtractor, MarkdownExtractor, YAMLExtractor, TOMLExtractor
)

EXT_MAP = {
    ".pdf": PDFExtractor,
    ".docx": DocxExtractor,
    ".csv": CSVExtractor,
    ".json": JSONExtractor,
    ".txt": TXTExtractor,
    ".md": MarkdownExtractor,
    ".markdown": MarkdownExtractor,
    ".yml": YAMLExtractor,
    ".yaml": YAMLExtractor,
    ".toml": TOMLExtractor,
}


def get_extractor(file_path: str):
    _, ext = os.path.splitext(file_path.lower())
    extractor_cls = EXT_MAP.get(ext)
    if not extractor_cls:
        raise ValueError(f"No extractor implemented for extension: {ext}")
    return extractor_cls(file_path)
