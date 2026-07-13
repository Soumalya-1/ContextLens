# Document Question Answering Assistant (RAG MVP)

A lightweight, beginner-friendly implementation of a Document Question Answering Assistant using Retrieval-Augmented Generation (RAG) with a Web Search Fallback. The project runs directly in your terminal, and fits entirely in under 350 lines of code.

## Tech Stack
- **Python**: Core programming language.
- **Groq SDK**: Fast LLM inference API (`openai/gpt-oss-20b` model).
- **sentence-transformers**: Generates dense vector embeddings (`all-MiniLM-L6-v2` model).
- **faiss-cpu**: Runs similarity searches over extracted document chunks.
- **pypdf**: Extracts text from PDF files.
- **duckduckgo-search**: Performs real-time web search fallback when the document similarity threshold is low.
- **python-dotenv**: Manages API keys safely.

## Project Structure
```text
DocQuery-AI/
├── app.py           # Orchestrates loading, indexing, and the interactive terminal CLI loop.
├── rag.py           # Embeddings generation, FAISS index creation, retrieval, and Groq LLM calls.
├── web_search.py    # DuckDuckGo fallback query execution.
├── utils.py         # PDF text extraction, document chunking, and sample PDF auto-generation.
├── requirements.txt # Project package dependencies.
├── .env.example     # Template for configuring API keys.
└── README.md        # Setup and usage guide.
```

## Features
1. **Auto-Generated Sample**: On startup, if no PDF is in the `data/` folder, the app generates a sample PDF (`data/sample.pdf`) containing definitions of RAG, FAISS, Sentence Transformers, and Groq to make the system runnable instantly.
2. **Text Processing**: Uses `pypdf` to read PDF pages, cleans whitespace, and splits it into overlapping 500-character chunks.
3. **FAISS Vector Store**: Normalizes sentence-transformer embeddings to perform cosine similarity searches using `IndexFlatIP`.
4. **Smart Threshold Routing**: 
   - If the similarity score is **above `0.35`**, the LLM answers using **only** document context (labeled `📄 Uploaded Document`).
   - If the score is **below `0.35`** (or no context is found), the system performs a live DuckDuckGo web search, and the LLM summarizes those search results (labeled `🌐 Web Search`).
   - If both document similarity and web search fail, a friendly fallback message is printed.
5. **Interactive Chat Loop**: Continues prompting for questions in the terminal until the user enters `exit`.

## Setup and Installation

### 1. Clone or Open the Workspace
Make sure your project files are structured as above.

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Setup Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
2. Open `.env` and fill in your Groq API key:
   ```text
   GROQ_API_KEY=gsk_your_groq_key_here
   ```
   *Get your key from [console.groq.com](https://console.groq.com/).*

### 4. Run the Application
Start the RAG chatbot:
```bash
python app.py
```

### 5. Exit
Type `exit` in the terminal to close the chatbot.
