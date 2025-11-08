# extractors/txt_extractor.py
from .base_extractor import BaseExtractor


class TXTExtractor(BaseExtractor):
    def extract_text(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
