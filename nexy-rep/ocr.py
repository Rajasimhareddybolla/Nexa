# ocr.py
import easyocr

# Initialize OCR reader (done once)
reader = easyocr.Reader(['en'], gpu=False)  # English, no GPU for local

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Extracted text.
    """
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return ' '.join(result)