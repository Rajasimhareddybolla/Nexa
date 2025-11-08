"""
Comprehensive test script for the Unified Service.
Tests both document extraction and image processing capabilities.
"""
import os
import time
from datetime import datetime
from services.services import UnifiedService

def test_document_extraction(service, test_files_dir):
    """Test document extraction for different file types."""
    print("\n=== Testing Document Extraction ===")
    
    # List of test files and their expected content checks
    test_cases = [
        {
            'file': 'test.txt',
            'description': 'Text file extraction',
            # expect the actual sample lines to appear
            'check': lambda text: 'This is a sample text file.' in text and 'We are testing text extraction.' in text
        },
        {
            'file': 'test.json',
            'description': 'JSON file extraction',
            'check': lambda text: '"name"' in text and '"age"' in text
        },
        {
            'file': 'test.yaml',
            'description': 'YAML file extraction',
            'check': lambda text: 'students' in text and 'Murali' in text
        },
        {
            'file': 'test.toml',
            'description': 'TOML file extraction',
            'check': lambda text: 'student1' in text and 'Murali' in text
        },
        {
            'file': 'test.md',
            'description': 'Markdown file extraction',
            'check': lambda text: 'Student Data' in text and 'Murali' in text
        }
    ]
    
    results = []
    for test_case in test_cases:
        file_path = os.path.join(test_files_dir, test_case['file'])
        if not os.path.exists(file_path):
            print(f"Skipping {test_case['file']} - file not found")
            continue
            
        print(f"\nTesting: {test_case['description']}")
        print(f"Input file: {file_path}")
        try:
            extracted_text = service.extract_from_file(file_path)
            success = test_case['check'](extracted_text)
            print("Status: Success" if success else "Status: Failed")
            print(f"Output preview: {extracted_text[:200]}...")
            results.append({
                'test': test_case['description'],
                'success': success,
                'output_length': len(extracted_text)
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                'test': test_case['description'],
                'success': False,
                'error': str(e)
            })
    
    return results

def test_image_processing(service):
    """Test image capture, OCR, and similarity detection."""
    print("\n=== Testing Image Processing ===")
    
    def print_result(operation, result):
        print(f"\nTesting: {operation}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Image path: {result['image_path']}")
        print(f"Text length: {len(result['text'])}")
        print(f"Text preview: {result['text'][:200]}...")
        print(f"Similarity score: {result['similarity']}")
    
    # Test 1: Initial screen capture
    print("\n1. Capturing initial screenshot")
    result1 = service.capture_and_process_screen()
    print_result("Initial capture", result1)
    
    if result1['image_path']:
        # Test 2: Process the same image (should show high similarity)
        print("\n2. Processing the same image (testing similarity detection)")
        result2 = service.process_image(result1['image_path'])
        print_result("Same image processing", result2)
        
        # Test 3: Wait and capture new screenshot (should show different content)
        print("\n3. Capturing new screenshot after delay")
        time.sleep(2)  # Wait for 2 seconds to ensure different screen content
        result3 = service.capture_and_process_screen()
        print_result("New capture", result3)
        
        results = {
            'initial_capture': {
                'success': bool(result1['text']),
                'has_image': bool(result1['image_path']),
                'text_length': len(result1['text'])
            },
            'similarity_test': {
                'success': result2['similarity'] > 0.9,  # Should be very similar
                'similarity_score': result2['similarity']
            },
            'different_capture': {
                'success': bool(result3['text']),
                'has_image': bool(result3['image_path']),
                'is_different': result3['similarity'] < result2['similarity']
            }
        }
    else:
        print("Initial capture failed - no image saved")
        results = {
            'initial_capture': {'success': False, 'error': 'No image saved'},
            'similarity_test': {'success': False, 'error': 'No initial image'},
            'different_capture': {'success': False, 'error': 'No initial image'}
        }
    
    return results

def main():
    # Initialize service
    service = UnifiedService()
    
    # Test document extraction
    test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
    doc_results = test_document_extraction(service, test_files_dir)
    
    # Test image processing
    img_results = test_image_processing(service)

    # Optional: Test GitHub activity if token is available
    print("\n=== GitHub Activity Test (optional) ===")
    github_token = os.getenv("GITHUB_TOKEN") 
    github_username = os.getenv("GITHUB_USERNAME") or "Rajasimhareddybolla"
    if github_token:
        # Use last 2 days by default
        from datetime import date, timedelta
        end = date.today()
        start = end - timedelta(days=2)
        # Pass all values as parameters (token can be passed directly)
        gh_summary = service.fetch_github_activity(
            username=github_username,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            token=github_token,
            repos=None,
        )
        print("GitHub summary:")
        print(gh_summary)
    else:
        print("Skipping GitHub tests: set GITHUB_TOKEN to enable API tests.")
    
    # Print summary
    print("\n=== Test Summary ===")
    print("\nDocument Extraction Results:")
    for result in doc_results:
        status = "✅ Passed" if result['success'] else "❌ Failed"
        print(f"{status} - {result['test']}")
    
    print("\nImage Processing Results:")
    for test, result in img_results.items():
        status = "✅ Passed" if result['success'] else "❌ Failed"
        print(f"{status} - {test}")
        if 'similarity_score' in result:
            print(f"  Similarity Score: {result['similarity_score']:.2f}")

if __name__ == "__main__":
    main()