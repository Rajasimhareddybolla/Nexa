# extractors/markdown_extractor.py
from .base_extractor import BaseExtractor
import markdown
import re


class MarkdownExtractor(BaseExtractor):
    def extract_text(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            md = f.read()
        # Convert markdown to HTML then strip tags to get plain text
        html = markdown.markdown(md)
        # Simple HTML tag stripper
        text = re.sub(r'<[^>]+>', '', html)
        return text.strip()
