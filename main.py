# main.py
import argparse
import os
from extractor_factory import get_extractor


def main():
    # Step 1: Create command-line argument parser
    parser = argparse.ArgumentParser(description="Universal text extractor")
    parser.add_argument("file", help="Path to input file")
    args = parser.parse_args()

    # Step 2: Verify file exists
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return

    # Step 3: Get appropriate extractor based on file type
    extractor = get_extractor(args.file)
    if not extractor:
        print(f"❌ Unsupported file format: {args.file}")
        return

    # Step 4: Extract text content
    print("🔍 Extracting text... please wait.")
    text = extractor.extract_text()

    # Step 5: Save extracted text to a .txt file
    base_name = os.path.splitext(args.file)[0]
    output_path = base_name + "_extracted.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Step 6: Print success message
    print("\n✅ Extraction completed successfully!")
    print(f"📄 Extracted text saved to: {output_path}\n")

    # (Optional) print first few lines for previ
    # 
    # ew
    print("Preview:")
    print("-" * 50)
    print(text[:500])  # shows first 500 characters
    print("-" * 50)


if __name__ == "__main__":
    main()
