# config.py
import os

class Config:
    """Configuration settings for the pipeline."""
    
    def __init__(self):
        # Directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_dir = os.path.join(self.base_dir, "images")
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.db_path = os.path.join(self.base_dir, "captured_data.db")
        
        # Create directories if not exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Pipeline settings
        self.interval_seconds = 30  # Change this to adjust interval
        self.similarity_threshold = 0.7  # 70%
        
        # Model settings
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        # For LangChain: HuggingFaceEmbeddings uses this model