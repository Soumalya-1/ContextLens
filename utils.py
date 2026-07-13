import os
import pypdf

def create_sample_pdf_if_missing(filepath: str):
    """
    Programmatically creates a sample PDF with informative content about RAG,
    FAISS, Sentence Transformers, and Groq. This ensures the app is runnable
    out of the box without requiring the user to supply a PDF first.
    """
    if os.path.exists(filepath):
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Facts and info to write to the PDF
    lines = [
        "Document Question Answering Assistant (RAG) MVP",
        "--------------------------------------------------",
        "Retrieval-Augmented Generation (RAG) combines document retrieval",
        "with a large language model to generate accurate answers.",
        "It reduces hallucinations and uses external knowledge without fine-tuning.",
        "",
        "FAISS (Facebook AI Similarity Search) is a library for efficient",
        "similarity search and clustering of dense vectors.",
        "",
        "Sentence Transformers is a library that generates state-of-the-art",
        "embeddings for sentences, paragraphs, and images. The all-MiniLM-L6-v2",
        "model is a popular choice for fast and accurate embeddings.",
        "",
        "Groq provides ultra-fast inference for large language models using its",
        "Tensor Streaming Processor (TSP) or Language Processing Unit (LPU)",
        "architecture, enabling real-time QA systems."
    ]
    
    # Construct standard minimal PDF content
    stream_parts = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        safe_line = line.replace("(", "\\(").replace(")", "\\)")
        stream_parts.append(f"({safe_line}) '")
    stream_parts.append("ET")
    stream_bytes = "\n".join(stream_parts).encode('latin-1')
    
    pdf_objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 595 842] /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode('latin-1') + stream_bytes + b"\nendstream\nendobj\n"
    ]
    
    with open(filepath, 'wb') as f:
        f.write(b"%PDF-1.4\n")
        offsets = []
        for obj in pdf_objs:
            offsets.append(f.tell())
            f.write(obj)
        
        xref_offset = f.tell()
        f.write(b"xref\n0 6\n0000000000 65535 f \n")
        for offset in offsets:
            f.write(f"{offset:010d} 00000 n \n".encode('latin-1'))
        
        f.write(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode('latin-1'))


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
