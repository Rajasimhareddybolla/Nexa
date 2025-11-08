"""
Demo script to showcase the unified service functionality.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from services.services import UnifiedService

def main():
    # Initialize the unified service
    service = UnifiedService()
    
    # Base directory for sample data
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    
    print("="*50)
    print("1. Testing Document Extraction")
    print("="*50)
    
    # Test text file extraction
    txt_path = os.path.join(sample_dir, "sample.txt")
    print("\nExtracting text from TXT file:")
    txt_content = service.extract_from_file(txt_path)
    print(f"Content:\n{txt_content}")
    
    # Test JSON file extraction
    json_path = os.path.join(sample_dir, "sample.json")
    print("\nExtracting text from JSON file:")
    json_content = service.extract_from_file(json_path)
    print(f"Content:\n{json_content}")
    
    print("\n" + "="*50)
    print("2. Testing Screen Capture and OCR")
    print("="*50)
    
    # Capture and process screen
    print("\nCapturing and processing screen:")
    screen_result = service.capture_and_process_screen()
    print(f"Timestamp: {screen_result['timestamp']}")
    print(f"Image saved: {screen_result['image_path']}")
    print(f"Extracted text:\n{screen_result['text']}")
    print(f"Similarity with previous: {screen_result['similarity']}")
    
    # Process the same image again to demonstrate similarity detection
    if screen_result['image_path']:
        print("\nProcessing the same image again (should show high similarity):")
        second_result = service.process_image(screen_result['image_path'])
        print(f"Similarity with previous: {second_result['similarity']}")
        print(f"Image saved: {second_result['image_path']}")  # Should be None due to similarity

if __name__ == "__main__":
    main()