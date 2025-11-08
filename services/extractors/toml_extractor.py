# extractors/toml_extractor.py
from .base_extractor import BaseExtractor
import json

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # for older versions


class TOMLExtractor(BaseExtractor):
    def extract_text(self) -> str:
        with open(self.file_path, 'rb') as f:
            data = tomllib.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
