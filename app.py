import os
import sys
from dotenv import load_dotenv
from groq import Groq

from utils import extract_text_from_pdf, split_text
from web_search import search_web
from rag import load_embedder, build_faiss_index, query_document, generate_groq_answer

def main():
    """
    Main entry point for the Document QA Assistant (RAG) CLI.
    Initializes environment, document processing, embedding model, and FAISS index,
    then enters the interactive chat loop.
    """
    # Load environment variables from .env
    load_dotenv()
    
    # Check for Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_key_here":
        print("\n[Error] GROQ_API_KEY is not set.")
        print("Please copy '.env.example' to '.env' and enter your Groq API key.")
        print("Obtain a free key at: https://console.groq.com/\n")
        sys.exit(1)
        
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        
    print("=" * 60)
    print("       Document Question Answering Assistant (RAG)")
    print("=" * 60)
    
    # Scan the data folder for PDF files
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF found. Please place a PDF inside the data folder.")
        sys.exit(1)
        
    if len(pdf_files) == 1:
        pdf_path = os.path.join(data_dir, pdf_files[0])
        print(f"[+] Loaded document: {pdf_path}")
    else:
        print("\nAvailable PDF documents:")
        for idx, filename in enumerate(pdf_files, 1):
            print(f"{idx}. {filename}")
        
        while True:
            try:
                choice = input("\nSelect a PDF to load (enter number): ").strip()
                if not choice:
                    continue
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(pdf_files):
                    pdf_path = os.path.join(data_dir, pdf_files[choice_idx])
                    print(f"[+] Loaded document: {pdf_path}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(pdf_files)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                sys.exit(0)
    
    # Extract text from the PDF
    print("[-] Extracting text from document...")
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print("[Error] No text could be extracted from the PDF.")
            sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to read PDF: {e}")
        sys.exit(1)
        
    # Split text into chunks
    print("[-] Splitting text into overlapping chunks...")
    chunks = split_text(raw_text, chunk_size=500, chunk_overlap=100)
    print(f"[+] Split document into {len(chunks)} chunks.")
    
    # Load embedding model
    print("[-] Loading sentence-transformers model ('all-MiniLM-L6-v2')...")
    embedder = load_embedder()
    
    # Generate embeddings and build FAISS index
    print("[-] Building FAISS similarity index...")
    index, indexed_chunks = build_faiss_index(chunks, embedder)
    print("[+] Similarity search index initialized successfully.")
    
    # Initialize Groq client
    groq_client = Groq(api_key=groq_api_key)
    
    print("\n[+] RAG MVP is ready! Type 'exit' to quit the application.")
    print("=" * 60)
    
    # Interactive chat loop
    while True:
        try:
            query = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
            
        if not query:
            continue
            
        if query.lower() == "exit":
            print("Exiting. Goodbye!")
            break
            
        print("\nThinking...")
        
        # Search the document database
        retrieved_chunks, max_score = query_document(
            query=query,
            index=index,
            chunks=indexed_chunks,
            model=embedder,
            threshold=similarity_threshold,
            top_k=3
        )
        
        # Check if matching context was found above threshold
        if retrieved_chunks:
            # Perform RAG completion using only document context
            context_text = "\n\n".join(retrieved_chunks)
            system_instruction = "Answer ONLY using the provided document context. Do not use outside knowledge."
            prompt = f"Document Context:\n{context_text}\n\nQuestion: {query}"
            
            answer = generate_groq_answer(groq_client, prompt, system_instruction, model_name=groq_model)
            
            print("\n📄 Uploaded Document")
            print(f"Similarity Score: {max_score:.4f}")
            print("-" * 30)
            print(answer)
            
        else:
            # Fall back to web search since document score was below threshold
            print(f"\n[Info] Similarity score ({max_score:.4f}) below threshold ({similarity_threshold:.2f}). Searching the web...")
            
            web_results = search_web(query)
            
            if web_results:
                # Perform LLM completion using only search snippets
                system_instruction = "Summarize the following web search results into a concise, factual answer. Mention if there are conflicting sources."
                prompt = f"Web Search Results:\n{web_results}\n\nQuestion: {query}"
                
                answer = generate_groq_answer(groq_client, prompt, system_instruction, model_name=groq_model)
                
                print("\n🌐 Web Search")
                print("-" * 30)
                print(answer)
            else:
                # Both document search and web fallback returned no results
                print("\nI couldn't find enough information in either the uploaded document or on the web.")

if __name__ == "__main__":
    main()
