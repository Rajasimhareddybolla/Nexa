# extractors/yaml_extractor.py
from .base_extractor import BaseExtractor
import yaml
import json


class YAMLExtractor(BaseExtractor):
    def extract_text(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Convert to pretty JSON-like text for readability
        return json.dumps(data, indent=2, ensure_ascii=False)
