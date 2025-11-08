# extractors/docx_extractor.py
from .base_extractor import BaseExtractor
from docx import Document


class DocxExtractor(BaseExtractor):
    def extract_text(self) -> str:
        doc = Document(self.file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs).strip()
