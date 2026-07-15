import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

def load_embedder() -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model used for generating embeddings.
    Using 'all-MiniLM-L6-v2' for its lightweight size and high speed.
    """
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def build_faiss_index(chunks: list, model: SentenceTransformer) -> tuple:
    """
    Generates embeddings for the list of chunks and creates a FAISS index.
    L2-normalization is applied to support Cosine Similarity using Inner Product (IndexFlatIP).
    """
    if not chunks:
        return None, []
        
    # Generate normalized embeddings (extract text if chunk is a dict)
    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Create Inner Product index (equivalent to Cosine Similarity for normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, chunks


def query_document(
    query: str, 
    index: faiss.IndexFlatIP, 
    chunks: list, 
    model: SentenceTransformer, 
    threshold: float = 0.15, 
    top_k: int = 3
) -> tuple:
    """
    Searches the FAISS index for relevant chunks.
    Returns (retrieved_chunks, max_score, top_hits).
    """
    try:
        if not index or not chunks:
            return [], 0.0, []
            
        # Generate query embedding and normalize
        query_emb = model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        
        # Retrieve at least 5 results (or len(chunks)) to get top 5 similarity scores for debugging
        k = min(max(top_k, 5), len(chunks))
        scores, indices = index.search(query_emb, k)
        
        # Extract the top score (highest cosine similarity)
        top_score = float(scores[0][0])
        
        top_hits = []
        for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                chunk_item = chunks[idx]
                if isinstance(chunk_item, dict):
                    chunk_text = chunk_item["text"]
                    meta = chunk_item
                else:
                    chunk_text = chunk_item
                    meta = None
                    
                top_hits.append({
                    "rank": rank_idx + 1,
                    "idx": int(idx),
                    "score": float(score),
                    "text": chunk_text,
                    "meta": meta
                })
        
        # Select chunks that are above the similarity threshold (up to top_k chunks)
        retrieved_chunks = []
        for hit in top_hits[:top_k]:
            if hit["score"] >= threshold:
                retrieved_chunks.append(hit["text"])
                
        return retrieved_chunks, top_score, top_hits
    except Exception as e:
        print(f"[query_document Error]: {e}")
        return [], 0.0, []




def generate_groq_answer(
    client: Groq, 
    prompt: str, 
    system_instruction: str, 
    model_name: str = "openai/gpt-oss-20b"
) -> str:
    """
    Queries the Groq API using the specified system instruction and prompt.
    Returns the generated answer.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Zero temperature for factual consistency
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err_msg = str(e)
        if any(term in err_msg.lower() for term in ["model_decommissioned", "model_not_found", "decommissioned", "not found"]):
            return (
                f"\n[Error] The configured Groq model '{model_name}' is no longer available (decommissioned or not found).\n"
                f"Please update 'GROQ_MODEL' in your '.env' file to a supported model."
            )
        return f"Error communicating with Groq API: {e}"


def save_faiss_index(index, chunks: list, index_path: str, chunks_path: str):
    """
    Saves the FAISS index and the associated text chunks to disk.
    """
    import os
    import json
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def load_faiss_index(index_path: str, chunks_path: str) -> tuple:
    """
    Loads the FAISS index and the associated text chunks from disk.
    """
    import json
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

