# extractors/base_extractor.py
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def extract_text(self) -> str:
        """Read file and return extracted plain text."""
        pass
