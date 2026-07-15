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


def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Extracts text page by page and chunks each page separately, preserving page metadata.
    Returns a list of dicts, each containing:
      - text: str
      - filename: str
      - chunk_id: int
      - page: int (1-indexed)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    filename = os.path.basename(pdf_path)
    chunks = []
    chunk_counter = 0
    
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            page_text = page.extract_text()
            if not page_text:
                continue
            
            # Clean whitespace
            cleaned_text = " ".join(page_text.split())
            if not cleaned_text:
                continue
                
            # Chunk the page text
            start = 0
            while start < len(cleaned_text):
                end = start + chunk_size
                chunk_text = cleaned_text[start:end]
                
                chunks.append({
                    "text": chunk_text,
                    "filename": filename,
                    "chunk_id": chunk_counter,
                    "page": page_num
                })
                chunk_counter += 1
                
                if end >= len(cleaned_text):
                    break
                start += chunk_size - chunk_overlap
                
    return chunks

