# extractors/pdf_extractor.py
from .base_extractor import BaseExtractor
import fitz  # PyMuPDF


class PDFExtractor(BaseExtractor):
    def extract_text(self) -> str:
        """Extract plain text from a PDF using PyMuPDF."""
        text_parts = []
        with fitz.open(self.file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts).strip()
