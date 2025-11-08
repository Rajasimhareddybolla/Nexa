"""
Unified service class that integrates Universal Extractor and Nexy-Rep functionalities.
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Import Universal Extractor classes
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DocxExtractor
from .extractors.csv_extractor import CSVExtractor
from .extractors.txt_extractor import TXTExtractor
from .extractors.json_extractor import JSONExtractor
from .extractors.yaml_extractor import YAMLExtractor
from .extractors.toml_extractor import TOMLExtractor
from .extractors.markdown_extractor import MarkdownExtractor

# Import Nexy-Rep functionalities
from .nexy_rep.capture import take_screenshot
from .nexy_rep.ocr import extract_text_from_image
from .nexy_rep.embed import get_embedding
from .nexy_rep.compare import compute_similarity
from .nexy_rep.storage import store_data, init_db
from .nexy_rep.config import Config
from .github_activity import GitHubUserActivity

class UnifiedService:
    """
    A unified service class that provides access to both document extraction
    and image processing/OCR capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified service.
        
        Args:
            config_path (str, optional): Path to the Nexy-Rep configuration file.
                If not provided, default configuration will be used.
        """
        # Initialize Nexy-Rep configuration
        self.config = Config(config_path) if config_path else Config()
        init_db(self.config.db_path)
        
        # File type to extractor mapping
        self.extractors = {
            '.pdf': PDFExtractor,
            '.docx': DocxExtractor,
            '.csv': CSVExtractor,
            '.txt': TXTExtractor,
            '.json': JSONExtractor,
            '.yaml': YAMLExtractor,
            '.yml': YAMLExtractor,
            '.toml': TOMLExtractor,
            '.md': MarkdownExtractor
        }
        
        # Keep track of last processed image for similarity comparison
        self._last_embedding = None
    
    def extract_from_file(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path (str): Path to the document file.
        
        Returns:
            str: Extracted text from the document.
            
        Raises:
            ValueError: If file type is not supported.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.extractors:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        extractor = self.extractors[file_ext](file_path)
        return extractor.extract_text()
    
    def capture_and_process_screen(self, store: bool = True) -> Dict[str, Any]:
        """
        Capture a screenshot, process it with OCR, and optionally store it.
        
        Args:
            store (bool): Whether to store the processed data. Defaults to True.
        
        Returns:
            dict: Dictionary containing the processed data:
                {
                    'timestamp': datetime object,
                    'image_path': str (path to stored image) or None,
                    'text': str (extracted text),
                    'similarity': float (similarity with previous image)
                }
        """
        # Generate timestamp and paths
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        temp_image_path = os.path.join(self.config.temp_dir, f"temp_{timestamp_str}.png")
        
        # Capture screenshot
        take_screenshot(temp_image_path)
        
        # Extract text using OCR
        text = extract_text_from_image(temp_image_path)
        
        result = {
            'timestamp': timestamp,
            'image_path': None,
            'text': text,
            'similarity': 0.0
        }
        
        if not text.strip():
            os.remove(temp_image_path)
            return result
        
        # Get embedding and compute similarity
        embedding = get_embedding(text)
        
        if self._last_embedding is not None:
            result['similarity'] = compute_similarity(embedding, self._last_embedding)
        
        if store and (self._last_embedding is None or 
                     result['similarity'] < self.config.similarity_threshold):
            # Store the image and data
            permanent_image_path = os.path.join(
                self.config.images_dir,
                f"image_{timestamp_str}.png"
            )
            os.rename(temp_image_path, permanent_image_path)
            
            store_data(
                self.config.db_path,
                timestamp=timestamp,
                image_path=permanent_image_path,
                extracted_text=text
            )
            
            result['image_path'] = permanent_image_path
            self._last_embedding = embedding
        else:
            os.remove(temp_image_path)
        
        return result
    
    def process_image(self, image_path: str, store: bool = True) -> Dict[str, Any]:
        """
        Process an existing image file with OCR and optionally store it.
        
        Args:
            image_path (str): Path to the image file.
            store (bool): Whether to store the processed data. Defaults to True.
            
        Returns:
            dict: Dictionary containing the processed data:
                {
                    'timestamp': datetime object,
                    'image_path': str (path to stored image) or None,
                    'text': str (extracted text),
                    'similarity': float (similarity with previous image)
                }
        """
        timestamp = datetime.now()
        
        # Extract text using OCR
        text = extract_text_from_image(image_path)
        
        result = {
            'timestamp': timestamp,
            'image_path': None,
            'text': text,
            'similarity': 0.0
        }
        
        if not text.strip():
            return result
        
        # Get embedding and compute similarity
        embedding = get_embedding(text)
        
        if self._last_embedding is not None:
            result['similarity'] = compute_similarity(embedding, self._last_embedding)
        
        if store and (self._last_embedding is None or 
                     result['similarity'] < self.config.similarity_threshold):
            # Store the data
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            permanent_image_path = os.path.join(
                self.config.images_dir,
                f"image_{timestamp_str}.png"
            )
            
            # Copy the image to permanent storage
            import shutil
            shutil.copy2(image_path, permanent_image_path)
            
            store_data(
                self.config.db_path,
                timestamp=timestamp,
                image_path=permanent_image_path,
                extracted_text=text
            )
            
            result['image_path'] = permanent_image_path
            self._last_embedding = embedding
        
        return result

    # ---------------------------------------------------------------
    # GitHub Activity Integration
    # ---------------------------------------------------------------
    def fetch_github_activity(
        self,
        username: str,
        start_date: str,
        end_date: str,
        token: Optional[str] = None,
        repos: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Fetch GitHub activity for a user in the given date range.

        Args:
            username: GitHub username
            start_date: Start date in 'YYYY-MM-DD'
            end_date: End date in 'YYYY-MM-DD'
            token: Optional GitHub token (falls back to GITHUB_TOKEN env var)
            repos: Optional list of repository names to filter

        Returns:
            dict: Summary returned by GitHubUserActivity.get_user_activity()
        """
        tracker = GitHubUserActivity(
            username=username,
            start_date=start_date,
            end_date=end_date,
            token=token,
            repos=repos,
        )

        return tracker.get_user_activity()
