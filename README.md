# ContextLens 🔍 — Website FAQ / Support Chatbot (RAG)

ContextLens is a modern, lightweight Website FAQ / Support Chatbot that uses **Retrieval-Augmented Generation (RAG)**. Built entirely in Python using **Streamlit** and **Groq**, it behaves like a chatbot for answering queries based on uploaded company documentation.

---

## 🚀 Key Features

- **Streamlit Interface**: Clean, ChatGPT-like conversation layout built for GitHub portfolio presentation.
- **Multi-Document Support**: Upload and manage multiple PDF files simultaneously, consolidating their contents into a single unified knowledge base.
- **Knowledge Base Management**: View, add, and delete documents in real-time from the sidebar, with stats on total chunks, docs, and last indexed time.
- **FAISS Vector Search**: Fast similarity search over all documents in the knowledge base using `IndexFlatIP` indexing.
- **SentenceTransformer Embeddings**: High-quality dense vector representations using `all-MiniLM-L6-v2`.
- **Groq LLM Integration**: Fast inference with the Groq client (`openai/gpt-oss-20b` or custom models).
- **Intelligent RAG Routing**: Instead of relying purely on brittle thresholds, ContextLens uses an LLM-based sufficiency auditor to evaluate context before answering.
- **Support Chatbot Fallback**: Returns a friendly "I don't know" message if the document context is insufficient, rather than guessing or searching the web.
- **Chat History**: Preserves conversation history locally across session reruns using `st.session_state`.
- **Debug Mode**: Toggleable metrics panel in the sidebar showing document analysis stats, top search hits, scores, and execution times (embedding, search, routing, and Groq).

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) for a clean, interactive user experience.
- **LLM API**: [Groq SDK](https://github.com/groq/groq-python) (running `openai/gpt-oss-20b` or custom models).
- **Embeddings**: [sentence-transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2` model, cached locally).
- **Vector DB**: [faiss-cpu](https://github.com/facebookresearch/faiss) for efficient dense vector similarity search.
- **PDF Extraction**: [pypdf](https://pypi.org/project/pypdf/) for reading and cleaning document text.

---

## 🏗️ Project Architecture

```text
ContextLens/
├── app.py           # Streamlit application (UI layout, state management, chat interaction)
├── rag.py           # Core RAG pipeline (index build, save/load persistence, Groq API client)
├── utils.py         # PDF text extraction and text chunking
├── requirements.txt # Project Python package dependencies
├── .env.example     # Configuration template for API keys and models
├── README.md        # System documentation and instructions
└── data/
    ├── documents/   # Persistent folder where uploaded PDF files are stored
    └── index/       # Index files directory
        ├── kb.index # Unified FAISS index containing vectors of all files
        ├── kb.chunks.json # Unified chunks list containing text and metadata
        ├── kb.meta.json # Knowledge Base metadata (docs count, chunks count, last indexed)
        └── cache/   # Cached chunk JSONs for individual PDF files (for incremental indexing)
```

---

## 🔄 Streamlit Workflow

```mermaid
graph TD
    A[Start ContextLens] --> B[Initialize Unified KB Index]
    B --> C{Scan data/documents/}
    C -->|Check Cache| D[Identify New/Deleted PDFs]
    D -->|Process New PDFs| E[Extract text page-by-page & chunk with metadata]
    E -->|Cache chunks| F[Save to data/index/cache/]
    F --> G[Aggregate all cached chunks of active PDFs]
    G --> H[Rebuild FAISS Vector Index once]
    H --> I[Save kb.index, kb.chunks.json & kb.meta.json]
    I --> J[Enable Chat Input]
    C -->|No changes| J
    J --> K[User asks Question]
    K --> L[Retrieve top 3 chunks from unified FAISS Index]
    L --> M[Ask Groq: Is context sufficient to answer?]
    M --> N{Decision YES?}
    N -->|Yes| O[Generate answer using Document Context]
    N -->|No / Index empty| P[Return 'I don't know' fallback message]
    O --> R[Add to Chat History st.session_state]
    P --> R
    R --> J
```

---

## ⚙️ How It Works

1. **Upload PDF**: PDF documents uploaded via the Streamlit interface are stored under `data/documents/`.
2. **Build Embeddings & Persistence**: The app parses the document text, splits it into overlapping 500-character chunks, embeds them using `all-MiniLM-L6-v2`, builds a FAISS index, and persists both index and chunks to `data/index/` for sub-second retrieval on future sessions.
3. **FAISS Retrieval**: When a query is asked, the system retrieves the **top 3** most similar chunks from the FAISS index (using a threshold of `-1.0` to ensure candidates are retrieved).
4. **LLM Context Evaluation**: The top 3 chunks are sent to the Groq LLM with an audit instruction. The LLM performs a sufficiency analysis and answers either `YES` or `NO` on whether the context is sufficient to answer the question.
5. **Answer Generation**:
   - If the audit returns **`YES`**, the LLM answers the query based **only** on the document chunks.
   - If the audit returns **`NO`** (or the index is empty), the chatbot returns: *"I don't know. I couldn't find that information in the uploaded company documentation."*

---

## 📦 Setup and Installation

### 1. Configure the Virtual Environment
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Configuration Variables
1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
2. Fill in your Groq API key and preferred model:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   SIMILARITY_THRESHOLD=0.15
   ```

---

## ⚙️ Environment Variables

ContextLens supports configuration via [`.env`](file:///C:/Projects/DocQuery-AI/.env):
- `GROQ_API_KEY`: API credential key from console.groq.com.
- `GROQ_MODEL`: Model Identifier (defaults to `openai/gpt-oss-20b`).
- `SIMILARITY_THRESHOLD`: The baseline similarity threshold for document chunks (defaults to `0.15`).

---

## 🏃 Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📸 User Interface Mockups

*Below are placeholder regions representing ContextLens in action:*

### 1. Document Upload & Selection (Sidebar)
`[=== Sidebar: Upload a PDF / Dropdown List of Available PDFs ===]`

### 2. Interactive Document QA (Document Source)
`[=== Chat Bubble (User): "What is FAISS?" ===]`  
`[=== Chat Bubble (AI): "FAISS is a library for similarity search..." ===]`  
`[=== Source Tag: 📄 Document | Score: 0.2084 ===]`

### 3. Chatbot Fallback (No Document Source)
`[=== Chat Bubble (User): "Who is the Prime Minister of Canada?" ===]`  
`[=== Chat Bubble (AI): "I don't know. I couldn't find that information in the uploaded company documentation." ===]`

---

## 🔮 Future Improvements
- **Custom System Instructions**: Let users configure system prompts directly in the Streamlit UI.
- **Document Formats**: Extend support to include `.txt`, `.docx`, and `.csv` files.
- **Offline Mode**: Add support for local vector search and LLMs via Ollama.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
