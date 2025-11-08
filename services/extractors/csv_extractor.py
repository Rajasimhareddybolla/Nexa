# extractors/csv_extractor.py
from .base_extractor import BaseExtractor
import pandas as pd


class CSVExtractor(BaseExtractor):
    def extract_text(self) -> str:
        # Read CSV and convert to readable text (rows as lines)
        df = pd.read_csv(self.file_path, dtype=str, keep_default_na=False)
        lines = []
        for _, row in df.iterrows():
            # join columns by tab or pipe to keep structure readable
            lines.append(" | ".join([str(v) for v in row.tolist()]))
        return "\n".join(lines).strip()
