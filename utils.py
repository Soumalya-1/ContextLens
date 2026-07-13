import os
import pypdf


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text contents from a given PDF file using PyPDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Splits continuous text into overlapping chunks of a given character size.
    """
    if not text:
        return []
        
    # Clean whitespace
    cleaned_text = " ".join(text.split())
    
    chunks = []
    start = 0
    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        # If we reached the end, break
        if end >= len(cleaned_text):
            break
        start += chunk_size - chunk_overlap
        
    return chunks
