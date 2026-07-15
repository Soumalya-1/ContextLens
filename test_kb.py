import os
import shutil
import json
import numpy as np
import faiss
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# We need to mock streamlit for the module level imports in app.py if needed,
# or we can test rag.py and utils.py functions directly.
# Since app.py contains streamlit calls, we can run a python process to test app.py,
# but for local unit/integration tests of the RAG pipeline, we can import rag and utils directly.

from utils import extract_and_chunk_pdf
from rag import load_embedder, build_faiss_index, query_document

DOCS_DIR = os.path.join("data", "documents")
INDEX_DIR = os.path.join("data", "index")

def test_pipeline():
    print("=== STARTING KNOWLEDGE BASE PIPELINE TESTS ===")
    
    # 1. Verify files exist in documents dir
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    print(f"Found PDF files for testing: {pdf_files}")
    if not pdf_files:
        print("ERROR: No PDF files found in data/documents/ directory to test with.")
        return
        
    # 2. Test extraction & chunking of the first file
    first_pdf = pdf_files[0]
    pdf_path = os.path.join(DOCS_DIR, first_pdf)
    print(f"\n--- Testing extract_and_chunk_pdf for '{first_pdf}' ---")
    chunks = extract_and_chunk_pdf(pdf_path)
    print(f"Successfully chunked '{first_pdf}'. Total chunks generated: {len(chunks)}")
    if chunks:
        print(f"Sample chunk 0: {json.dumps(chunks[0], indent=2)}")
        assert "text" in chunks[0]
        assert "filename" in chunks[0]
        assert "chunk_id" in chunks[0]
        assert "page" in chunks[0]
        print("[OK] Chunk metadata fields verified.")
    else:
        print("WARNING: First PDF returned 0 chunks (might be scanned or empty).")

    # 3. Load cached SentenceTransformer embedder
    print("\n--- Loading cached SentenceTransformer embedder ---")
    embedder = load_embedder()
    print("[OK] Model loaded successfully.")

    # 4. Test unified index build with multiple files (if available)
    print("\n--- Building Unified Knowledge Base Index ---")
    all_chunks = []
    for pdf in pdf_files:
        path = os.path.join(DOCS_DIR, pdf)
        try:
            file_chunks = extract_and_chunk_pdf(path)
            all_chunks.extend(file_chunks)
            print(f"Processed '{pdf}': {len(file_chunks)} chunks added.")
        except Exception as e:
            print(f"Failed to process '{pdf}': {e}")
            
    print(f"Total chunks in unified Knowledge Base: {len(all_chunks)}")
    
    if not all_chunks:
        print("FAIL: No chunks extracted across any document.")
        return
        
    index, indexed_chunks = build_faiss_index(all_chunks, embedder)
    print(f"[OK] FAISS Index built. Total vectors: {index.ntotal if index else 0}")
    assert index is not None
    assert len(indexed_chunks) == len(all_chunks)

    # 5. Query across all documents
    print("\n--- Querying Unified Knowledge Base ---")
    # Let's search for a generic question or use keywords from the filenames
    query_text = "What is the content of the documents?"
    retrieved, score, hits = query_document(
        query=query_text,
        index=index,
        chunks=indexed_chunks,
        model=embedder,
        threshold=-1.0,
        top_k=3
    )
    print(f"Query: '{query_text}'")
    print(f"Top similarity score: {score:.4f}")
    print(f"Retrieved {len(retrieved)} chunks.")
    
    for idx, hit in enumerate(hits[:3]):
        meta = hit.get("meta")
        print(f"  Hit {idx+1}: Score: {hit['score']:.4f}")
        if isinstance(meta, dict):
            print(f"    Document: {meta.get('filename')}")
            print(f"    Chunk ID: {meta.get('chunk_id')}")
            print(f"    Page: {meta.get('page')}")
        else:
            print(f"    No metadata (Legacy chunk)")
        print(f"    Snippet: {hit['text'][:120]}...")
        
    # Verify that metadata matches and is correct
    assert len(hits) > 0
    if len(all_chunks) > 0:
        first_hit = hits[0]
        assert "meta" in first_hit
        assert first_hit["meta"] is not None
        assert "filename" in first_hit["meta"]
        assert "chunk_id" in first_hit["meta"]
        
    print("\n=== PIPELINE TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_pipeline()
