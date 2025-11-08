# main.py
import time
import logging
import os
from datetime import datetime
from capture import take_screenshot
from ocr import extract_text_from_image
from embed import get_embedding
from compare import compute_similarity
from storage import store_data, init_db
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main pipeline loop."""
    config = Config()
    init_db(config.db_path)
    
    last_embedding = None
    last_image_path = None  # To clean up discarded images
    
    logging.info("Starting the image capture and processing pipeline.")
    logging.info(f"Interval: {config.interval_seconds} seconds")
    logging.info(f"Similarity threshold: {config.similarity_threshold}")
    
    try:
        while True:
            # Step 1: Capture screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_image_path = os.path.join(config.temp_dir, f"temp_{timestamp}.png")
            take_screenshot(temp_image_path)
            logging.info(f"Captured screenshot: {temp_image_path}")
            
            # Step 2: Extract text using OCR
            text = extract_text_from_image(temp_image_path)
            if not text.strip():
                logging.warning("No text extracted from image. Discarding.")
                os.remove(temp_image_path)
                time.sleep(config.interval_seconds)
                continue
            
            logging.info(f"Extracted text (first 100 chars): {text[:100]}...")
            
            # Step 3: Get embedding
            embedding = get_embedding(text)
            
            # Step 4: Compare with previous
            if last_embedding is None:
                similarity = 0.0  # First image, always store
            else:
                similarity = compute_similarity(embedding, last_embedding)
            
            logging.info(f"Similarity with previous: {similarity:.2f}")
            
            if similarity < config.similarity_threshold:
                # Store
                permanent_image_path = os.path.join(
                    config.images_dir,
                    f"image_{timestamp}.png"
                )
                os.rename(temp_image_path, permanent_image_path)
                store_data(
                    config.db_path,
                    timestamp=datetime.now(),
                    image_path=permanent_image_path,
                    extracted_text=text
                )
                logging.info(f"Stored new data: {permanent_image_path}")
                
                # Update last
                last_embedding = embedding
            else:
                # Discard
                os.remove(temp_image_path)
                logging.info("Image discarded due to high similarity.")
            
            # Wait for next interval
            time.sleep(config.interval_seconds)
    
    except KeyboardInterrupt:
        logging.info("Pipeline stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()