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
        
    # Generate normalized embeddings
    embeddings = model.encode(chunks, normalize_embeddings=True)
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
    threshold: float = 0.35, 
    top_k: int = 3
) -> tuple:
    """
    Searches the FAISS index for relevant chunks.
    If the highest similarity score meets the threshold, returns top_k chunks and their scores.
    Otherwise, returns (None, score) indicating search fell below threshold.
    """
    if not index or not chunks:
        return None, 0.0
        
    # Generate query embedding and normalize
    query_emb = model.encode([query], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")
    
    # Retrieve top_k results
    k = min(top_k, len(chunks))
    scores, indices = index.search(query_emb, k)
    
    # Extract the top score (highest cosine similarity)
    top_score = float(scores[0][0])
    
    if top_score >= threshold:
        retrieved_chunks = [chunks[idx] for idx in indices[0] if idx != -1]
        return retrieved_chunks, top_score
        
    return None, top_score


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
