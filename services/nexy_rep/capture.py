# capture.py
import pyautogui

def take_screenshot(filename: str) -> None:
    """
    Capture a screenshot and save to the given filename.
    
    Args:
        filename (str): Path to save the screenshot.
    """
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)