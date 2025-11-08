# extractors/json_extractor.py
from .base_extractor import BaseExtractor
import json


class JSONExtractor(BaseExtractor):
    def extract_text(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Pretty-print JSON as text
        return json.dumps(data, indent=2, ensure_ascii=False)
