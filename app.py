import os
import sys
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils import extract_text_from_pdf, split_text
from web_search import search_web
from rag import (
    load_embedder,
    build_faiss_index,
    query_document,
    generate_groq_answer,
    save_faiss_index,
    load_faiss_index
)

# Load environment variables
load_dotenv()

# Streamlit Page Configurations
st.set_page_config(
    page_title="ContextLens - Document QA Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Title styling */
    .title-text {
        background: linear-gradient(135deg, #FF3366, #FF9933, #33CCFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    
    /* Metadata bubble styling */
    .metadata-bubble {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 0.85rem;
        display: inline-flex;
        gap: 15px;
        color: #888;
    }
    
    .source-tag-doc {
        color: #4CAF50;
        font-weight: 600;
    }
    
    .source-tag-web {
        color: #00bcd4;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Configuration Setup
groq_api_key = os.getenv("GROQ_API_KEY")
similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Directories setup
DOCS_DIR = os.path.join("data", "documents")
INDEX_DIR = os.path.join("data", "index")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_times" not in st.session_state:
    st.session_state.debug_times = {"embedding": 0.0, "faiss_search": 0.0, "groq_api": 0.0, "web_search": 0.0}
if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "last_uploaded_processed" not in st.session_state:
    st.session_state.last_uploaded_processed = None

# Validate API Key
if not groq_api_key or groq_api_key == "your_key_here":
    st.error("❌ **GROQ_API_KEY is not set.** Please configure it in your `.env` file to start using the assistant.")
    st.stop()

# Initialize API client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=groq_api_key)

groq_client = get_groq_client()

# Cache the heavy SentenceTransformer model
@st.cache_resource
def get_embedder():
    return load_embedder()

# Sidebar Layout
st.sidebar.markdown("<h2 style='margin-bottom: 0px;'>ContextLens 🔍</h2>", unsafe_allow_html=True)
st.sidebar.caption("Document QA Assistant with Web Fallback")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a new PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_name = uploaded_file.name
    # Save and update active doc only if we haven't processed this specific file upload yet
    if st.session_state.get("last_uploaded_processed") != pdf_name:
        pdf_path = os.path.join(DOCS_DIR, pdf_name)
        file_exists = os.path.exists(pdf_path)
        
        try:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if file_exists:
                st.sidebar.info(f"🔄 Overwrote existing document: {pdf_name}")
            else:
                st.sidebar.success(f"✅ Saved new document: {pdf_name}")
                
            st.session_state.last_uploaded_processed = pdf_name
            st.session_state.active_doc = pdf_name
            st.session_state.loaded_doc_name = None  # Force reload/reindex
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to write uploaded file: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Select Document")

# Get list of existing PDF files
pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])

# Initialize active_doc if not set but files exist
if "active_doc" not in st.session_state:
    if pdf_files:
        st.session_state.active_doc = pdf_files[0]
    else:
        st.session_state.active_doc = None

if pdf_files:
    # Set selectbox default index based on active_doc
    default_idx = 0
    if st.session_state.active_doc in pdf_files:
        default_idx = pdf_files.index(st.session_state.active_doc)
        
    selected_pdf = st.sidebar.selectbox(
        "Available PDFs",
        options=pdf_files,
        index=default_idx,
        key="active_doc_selector"
    )
    
    # Update active doc state if selectbox changes
    if st.session_state.active_doc != selected_pdf:
        st.session_state.active_doc = selected_pdf
        st.session_state.loaded_doc_name = None  # Force reload/reindex
        st.rerun()
else:
    st.sidebar.info("No documents uploaded yet. Please upload a PDF to get started.")
    st.session_state.active_doc = None

# Sidebar Debug Option
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🐞 Debug Mode", value=False)

# Index processing for the selected document
active_doc = st.session_state.get("active_doc")

if active_doc:
    pdf_path = os.path.join(DOCS_DIR, active_doc)
    index_path = os.path.join(INDEX_DIR, f"{active_doc}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{active_doc}.chunks.json")
    
    # If the index is already built on disk, load it
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        if st.session_state.get("loaded_doc_name") != active_doc:
            with st.spinner(f"Loading search index for {active_doc}..."):
                import time
                start_t = time.time()
                try:
                    index, chunks = load_faiss_index(index_path, chunks_path)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.loaded_doc_name = active_doc
                    st.session_state.debug_times["embedding"] = time.time() - start_t
                except Exception as e:
                    st.sidebar.error(f"Error loading index: {e}. Rebuilding...")
                    st.session_state.loaded_doc_name = None
                    
    # Rebuild index if missing or failed to load
    if st.session_state.get("loaded_doc_name") != active_doc:
        with st.spinner(f"Analyzing and indexing {active_doc}..."):
            import time
            start_t = time.time()
            try:
                # Extract text
                raw_text = extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    st.error(f"❌ **Error**: No readable text could be extracted from '{active_doc}'.")
                    st.session_state.active_doc = None
                    st.session_state.index = None
                    st.session_state.chunks = None
                    st.stop()
                
                # Split text into overlapping chunks
                chunks = split_text(raw_text, chunk_size=500, chunk_overlap=100)
                
                # Initialize embedder and build FAISS index
                embedder = get_embedder()
                index, indexed_chunks = build_faiss_index(chunks, embedder)
                
                # Save index and metadata
                save_faiss_index(index, indexed_chunks, index_path, chunks_path)
                
                st.session_state.index = index
                st.session_state.chunks = indexed_chunks
                st.session_state.loaded_doc_name = active_doc
                st.session_state.debug_times["embedding"] = time.time() - start_t
                st.sidebar.success(f"⚡ Indexed {active_doc} successfully!")
            except Exception as e:
                import traceback
                st.error(f"❌ **Failed to process PDF**: {e}")
                st.text(traceback.format_exc())
                st.session_state.active_doc = None
                st.session_state.index = None
                st.session_state.chunks = None
                st.stop()

# Sidebar Debug Information display
if debug_mode:
    st.sidebar.markdown("### 🐞 Debug Metrics")
    st.sidebar.write(f"**Active Document**: `{active_doc}`")
    st.sidebar.write(f"**Loaded in Memory**: `{st.session_state.get('loaded_doc_name')}`")
    st.sidebar.write(f"**Number of Chunks**: `{len(st.session_state.get('chunks', []))}`")
    st.sidebar.write(f"**Similarity Score**: `{st.session_state.get('last_score', 0.0):.4f}`")
    
    # Times
    times = st.session_state.get("debug_times", {})
    st.sidebar.write("**Time Taken**:")
    st.sidebar.write(f"- Embedding/Index Load: `{times.get('embedding', 0.0):.4f}s`")
    st.sidebar.write(f"- FAISS search: `{times.get('faiss_search', 0.0):.4f}s`")
    st.sidebar.write(f"- Groq API: `{times.get('groq_api', 0.0):.4f}s`")
    st.sidebar.write(f"- Web search: `{times.get('web_search', 0.0):.4f}s`")
    
    with st.sidebar.expander("Retrieved Chunks Details", expanded=False):
        chunks_list = st.session_state.get("last_retrieved_chunks", [])
        if chunks_list:
            for idx, c in enumerate(chunks_list, 1):
                st.caption(f"**Chunk {idx}**:")
                st.text(c)
                st.markdown("---")
        else:
            st.caption("No chunks retrieved in last query.")

# Main UI Area
st.markdown("<h1 class='title-text'>ContextLens 🔍</h1>", unsafe_allow_html=True)
if active_doc:
    st.markdown(f"Currently querying: **`{active_doc}`**")
else:
    st.markdown("No document loaded.")

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("source") and msg["source"] != "None":
            source_tag = msg["source"]
            tag_class = "source-tag-doc" if "Document" in source_tag else "source-tag-web"
            st.markdown(
                f"<div class='metadata-bubble'>"
                f"<span>Source: <span class='{tag_class}'>{source_tag}</span></span>"
                f"<span>Similarity Score: <b>{msg['score']:.4f}</b></span>"
                f"<span>Document: <b>{msg.get('document', 'N/A')}</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )

# Clear History button in sidebar
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Disable input if no active doc is selected
chat_disabled = (active_doc is None)
if chat_disabled:
    st.warning("⚠️ **Please upload a PDF or select an existing document from the sidebar to start asking questions.**")

query = st.chat_input("Ask a question about the active document...", disabled=chat_disabled)

if query:
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
        
    with st.chat_message("assistant"):
        import time
        import traceback
        
        answer = ""
        source = "None"
        max_score = 0.0
        doc_name = "N/A"
        retrieved_chunks = []
        
        # Reset query-specific times
        st.session_state.debug_times["faiss_search"] = 0.0
        st.session_state.debug_times["groq_api"] = 0.0
        st.session_state.debug_times["web_search"] = 0.0
        
        print("\n=== PIPELINE RUN ===")
        # We will use st.status to show the pipeline steps
        with st.status("🔍 Processing Query through RAG Pipeline...", expanded=True) as status_box:
            try:
                # [STEP 1] Question received
                status_box.write("📥 **[STEP 1] Question received**: " + query)
                print(f"[STEP 1] Question received: {query}")
                
                # [STEP 2] Active document
                status_box.write(f"📄 **[STEP 2] Active document**: {active_doc}")
                print(f"[STEP 2] Active document: {active_doc}")
                
                # [STEP 3] Loading FAISS index
                # Index is validated and loaded globally, so we log its status here
                status_box.write(f"📦 **[STEP 3] FAISS index status**: Ready for {active_doc} (contains {len(st.session_state.get('chunks', []))} chunks)")
                print(f"[STEP 3] FAISS index status: Ready for {active_doc}")
                
                # [STEP 4] Running similarity search
                status_box.write("🔎 **[STEP 4] Running similarity search** over document chunks...")
                print("[STEP 4] Running similarity search...")
                
                start_t = time.time()
                embedder = get_embedder()
                retrieved_chunks, max_score = query_document(
                    query=query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    model=embedder,
                    threshold=similarity_threshold,
                    top_k=3
                )
                st.session_state.debug_times["faiss_search"] = time.time() - start_t
                st.session_state.last_score = max_score
                st.session_state.last_retrieved_chunks = retrieved_chunks
                
                # [STEP 5] Similarity score
                status_box.write(f"📊 **[STEP 5] Similarity score**: {max_score:.4f} (Threshold: {similarity_threshold:.2f})")
                print(f"[STEP 5] Similarity score: {max_score:.4f}")
                
                # RAG or Web Fallback
                if retrieved_chunks:
                    source = "📄 Document"
                    doc_name = active_doc
                    context_text = "\n\n".join(retrieved_chunks)
                    system_instruction = "Answer ONLY using the provided document context. Do not use outside knowledge."
                    prompt = f"Document Context:\n{context_text}\n\nQuestion: {query}"
                    
                    # [STEP 6] Calling Groq
                    status_box.write(f"🤖 **[STEP 6] Calling Groq** using model `{groq_model}`...")
                    print(f"[STEP 6] Calling Groq with model: {groq_model}")
                    
                    start_t = time.time()
                    answer = generate_groq_answer(groq_client, prompt, system_instruction, model_name=groq_model)
                    st.session_state.debug_times["groq_api"] = time.time() - start_t
                else:
                    status_box.write("🌐 **[STEP 5a] Similarity score below threshold**. Falling back to Web Search.")
                    print("[STEP 5a] Below threshold, fallback to Web Search.")
                    
                    # [STEP 7] Running web search
                    status_box.write("🕸️ **[STEP 7] Running web search** via DuckDuckGo fallback...")
                    print(f"[STEP 7] Running web search for query: {query}")
                    
                    start_t = time.time()
                    web_results = search_web(query)
                    st.session_state.debug_times["web_search"] = time.time() - start_t
                    
                    if web_results and not web_results.startswith("Error:"):
                        source = "🌐 Web Search"
                        system_instruction = "Summarize the following web search results into a concise, factual answer. Mention if there are conflicting sources."
                        prompt = f"Web Search Results:\n{web_results}\n\nQuestion: {query}"
                        
                        status_box.write("🤖 **[STEP 7a] Calling Groq** to summarize web results...")
                        print("[STEP 7a] Calling Groq for web results summarization...")
                        
                        start_t = time.time()
                        answer = generate_groq_answer(groq_client, prompt, system_instruction, model_name=groq_model)
                        st.session_state.debug_times["groq_api"] = time.time() - start_t
                    elif web_results.startswith("Error:"):
                        answer = f"Error: Web search timed out or failed. ({web_results})"
                        source = "None"
                    else:
                        answer = "I couldn't find enough information in either the uploaded document or on the web."
                        source = "None"
                        
                # [STEP 8] Returning response
                status_box.write("📤 **[STEP 8] Returning response** to user interface.")
                print("[STEP 8] Returning response.")
                
                status_box.update(label="✅ Processing Complete!", state="complete", expanded=False)
                
            except Exception as pipeline_err:
                status_box.update(label="❌ Pipeline Failed", state="error", expanded=True)
                st.error(f"### ❌ Pipeline Failure during processing")
                st.write(f"**Error Details**: {pipeline_err}")
                st.text(traceback.format_exc())
                print(f"[PIPELINE ERROR]: {pipeline_err}")
                print(traceback.format_exc())
                answer = f"Error: A pipeline processing failure occurred. Details: {pipeline_err}"
                source = "None"
                
        # Display response
        if answer.startswith("Error") or "[Error]" in answer:
            st.error(answer)
        else:
            st.write(answer)
            
        if source != "None":
            tag_class = "source-tag-doc" if "Document" in source else "source-tag-web"
            st.markdown(
                f"<div class='metadata-bubble'>"
                f"<span>Source: <span class='{tag_class}'>{source}</span></span>"
                f"<span>Similarity Score: <b>{max_score:.4f}</b></span>"
                f"<span>Document: <b>{doc_name}</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source": source,
            "score": max_score,
            "document": doc_name
        })
        
        # Force immediate rerun to render updated chat history correctly
        st.rerun()
