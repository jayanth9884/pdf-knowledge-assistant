from pypdf import PdfReader

pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- Page {page_number} ---")

    if text:
        print(text)
    else:
        print("No extractable text found.")