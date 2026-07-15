import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils import extract_text_from_pdf, split_text, extract_and_chunk_pdf
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
similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
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
    st.session_state.debug_times = {"embedding": 0.0, "faiss_search": 0.0, "rag_routing": 0.0, "groq_api": 0.0}
if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "last_query_top_hits" not in st.session_state:
    st.session_state.last_query_top_hits = []
if "last_router_decision" not in st.session_state:
    st.session_state.last_router_decision = "N/A"
if "debug_doc_info" not in st.session_state:
    st.session_state.debug_doc_info = {}
if "last_uploaded_processed" not in st.session_state:
    st.session_state.last_uploaded_processed = []
if "admin_messages" not in st.session_state:
    st.session_state.admin_messages = []

# Validate API Key
if not groq_api_key or groq_api_key == "your_key_here":
    st.error("❌ **GROQ_API_KEY is not set.** Please configure it in your `.env` file to start using the assistant.")
    st.stop()

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=groq_api_key)

groq_client = get_groq_client()

# Cache the SentenceTransformer model
@st.cache_resource
def get_embedder():
    return load_embedder()

# Helper function to gather and print document stats for Debug Mode
def populate_debug_doc_info(index, chunks):
    try:
        avg_sz = sum(len(c["text"] if isinstance(c, dict) else c) for c in chunks) / len(chunks) if chunks else 0
        st.session_state.debug_doc_info = {
            "num_chunks": len(chunks),
            "avg_chunk_size": avg_sz,
            "dimension": index.d if index else 0,
            "total_vectors": index.ntotal if index else 0
        }
        # Print diagnostic stats to terminal
        print("\n=== [DIAGNOSTIC PIPELINE STATE: INDEX] ===")
        print(f"1. Chunking: number of chunks = {len(chunks)}, average chunk size = {avg_sz:.2f}")
        print(f"2. Embeddings: dimension = {index.d if index else 0}, vectors = {index.ntotal if index else 0}")
        print(f"3. FAISS: total vectors stored = {index.ntotal if index else 0}")
        print("==========================================\n")
    except Exception as e:
        print(f"[populate_debug_doc_info Error]: {e}")

def load_kb_metadata():
    import json
    meta_path = os.path.join(INDEX_DIR, "kb.meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_indexed": "N/A", "total_documents": 0, "total_chunks": 0}

def rebuild_knowledge_base(force: bool = False):
    import time
    from datetime import datetime
    import json
    
    os.makedirs(os.path.join(INDEX_DIR, "cache"), exist_ok=True)
    pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])
    
    if force:
        cache_dir = os.path.join(INDEX_DIR, "cache")
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith(".chunks.json"):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except Exception as e:
                        print(f"Failed to delete cache file {f}: {e}")
                        
    all_chunks = []
    embedder = get_embedder()
    
    for pdf in pdf_files:
        pdf_path = os.path.join(DOCS_DIR, pdf)
        cache_path = os.path.join(INDEX_DIR, "cache", f"{pdf}.chunks.json")
        
        if not os.path.exists(cache_path):
            try:
                file_chunks = extract_and_chunk_pdf(pdf_path)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(file_chunks, f, ensure_ascii=False)
            except Exception as e:
                print(f"Error processing {pdf}: {e}")
                continue
        else:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    file_chunks = json.load(f)
            except Exception as e:
                print(f"Error reading cache for {pdf}: {e}")
                try:
                    file_chunks = extract_and_chunk_pdf(pdf_path)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(file_chunks, f, ensure_ascii=False)
                except Exception as e2:
                    print(f"Failed to reprocess {pdf}: {e2}")
                    continue
                    
        all_chunks.extend(file_chunks)
        
    if all_chunks:
        try:
            index_path = os.path.join(INDEX_DIR, "kb.index")
            chunks_path = os.path.join(INDEX_DIR, "kb.chunks.json")
            
            index, indexed_chunks = build_faiss_index(all_chunks, embedder)
            save_faiss_index(index, indexed_chunks, index_path, chunks_path)
            
            meta_path = os.path.join(INDEX_DIR, "kb.meta.json")
            meta_data = {
                "last_indexed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(pdf_files),
                "total_chunks": len(all_chunks)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False)
                
            st.session_state.index = index
            st.session_state.chunks = indexed_chunks
            populate_debug_doc_info(index, indexed_chunks)
        except Exception as e:
            print(f"Error building FAISS index: {e}")
    else:
        st.session_state.index = None
        st.session_state.chunks = []
        index_path = os.path.join(INDEX_DIR, "kb.index")
        chunks_path = os.path.join(INDEX_DIR, "kb.chunks.json")
        meta_path = os.path.join(INDEX_DIR, "kb.meta.json")
        for p in [index_path, chunks_path, meta_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

def get_file_size_str(filepath: str) -> str:
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        return "N/A"

def display_admin_messages():
    if "admin_messages" in st.session_state and st.session_state.admin_messages:
        for msg in st.session_state.admin_messages:
            if msg["type"] == "success":
                st.sidebar.success(msg["text"])
            elif msg["type"] == "warning":
                st.sidebar.warning(msg["text"])
            elif msg["type"] == "error":
                st.sidebar.error(msg["text"])
        st.session_state.admin_messages = []

def delete_document(pdf_name: str):
    pdf_path = os.path.join(DOCS_DIR, pdf_name)
    success = False
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            success = True
        except Exception as e:
            st.session_state.admin_messages.append({"type": "error", "text": f"Failed to delete {pdf_name}: {e}"})
            
    cache_path = os.path.join(INDEX_DIR, "cache", f"{pdf_name}.chunks.json")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception as e:
            print(f"Failed to delete cache file {cache_path}: {e}")
            
    rebuild_knowledge_base()
    if success:
        st.session_state.admin_messages.append({"type": "success", "text": f"✅ Document deleted."})

def format_sources(retrieved_hits) -> str:
    citation_strings = []
    seen = set()
    for hit in retrieved_hits:
        meta = hit.get("meta")
        if isinstance(meta, dict):
            filename = meta.get("filename", "N/A")
            chunk_id = meta.get("chunk_id", "N/A")
            page = meta.get("page")
            
            key = (filename, chunk_id, page)
            if key in seen:
                continue
            seen.add(key)
            
            cit = f"Source:\nDocument: {filename}\nChunk: {chunk_id}"
            if page is not None:
                cit += f"\nPage: {page}"
            citation_strings.append(cit)
        elif isinstance(hit, dict) and "text" in hit:
            doc_name = st.session_state.get("active_doc", "N/A")
            idx = hit.get("idx", "N/A")
            cit = f"Source:\nDocument: {doc_name}\nChunk: {idx}"
            citation_strings.append(cit)
            
    return "\n\n".join(citation_strings)

# Startup/initialization of the KB index
index_path = os.path.join(INDEX_DIR, "kb.index")
chunks_path = os.path.join(INDEX_DIR, "kb.chunks.json")

# Determine if we need to load or initialize the index
if "index" not in st.session_state or "chunks" not in st.session_state:
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            index, chunks = load_faiss_index(index_path, chunks_path)
            st.session_state.index = index
            st.session_state.chunks = chunks
            populate_debug_doc_info(index, chunks)
        except Exception as e:
            print(f"Failed to load FAISS index on startup: {e}")
            rebuild_knowledge_base()
    else:
        pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])
        if pdf_files:
            rebuild_knowledge_base()
        else:
            st.session_state.index = None
            st.session_state.chunks = []
pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])

# Sidebar Layout
st.sidebar.markdown("<h2 style='margin-bottom: 0px;'>ContextLens 🔍</h2>", unsafe_allow_html=True)
st.sidebar.caption("Website FAQ / Support Chatbot")
st.sidebar.markdown("---")

# Mode selection toggle
mode = st.sidebar.radio(
    "👤 Mode",
    options=["Visitor", "Admin"],
    index=0,
    help="Select 'Visitor' to chat with the FAQ bot, or 'Admin' to manage documentation."
)
st.sidebar.markdown("---")

if mode == "Admin":
    st.sidebar.markdown("### 🛠️ Admin Dashboard")
    
    # Display any success / warning notifications from actions
    display_admin_messages()
    
    # 1. Upload Documents Expander
    with st.sidebar.expander("📂 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or multiple PDF documents to add them to the knowledge base.",
            key="admin_pdf_uploader"
        )
        overwrite = st.checkbox("Overwrite existing files", value=False, key="overwrite_files")
        
        # Process uploaded files
        if uploaded_files:
            uploaded_names = [f.name for f in uploaded_files]
            last_processed = st.session_state.get("last_uploaded_processed", [])
            
            if uploaded_names != last_processed:
                new_files_uploaded = False
                skipped_files = []
                uploaded_success_files = []
                
                for uploaded_file in uploaded_files:
                    pdf_name = uploaded_file.name
                    pdf_path = os.path.join(DOCS_DIR, pdf_name)
                    file_exists = os.path.exists(pdf_path)
                    
                    if file_exists and not overwrite:
                        skipped_files.append(pdf_name)
                        continue
                        
                    try:
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Delete cache for this file to ensure it's re-indexed
                        cache_path = os.path.join(INDEX_DIR, "cache", f"{pdf_name}.chunks.json")
                        if os.path.exists(cache_path):
                            try:
                                os.remove(cache_path)
                            except Exception:
                                pass
                        new_files_uploaded = True
                        uploaded_success_files.append(pdf_name)
                    except Exception as e:
                        st.error(f"Failed to save {pdf_name}: {e}")
                
                st.session_state.last_uploaded_processed = uploaded_names
                
                if skipped_files:
                    st.session_state.admin_messages.append({
                        "type": "warning", 
                        "text": f"⚠️ Skipped duplicate(s): {', '.join(skipped_files)}. Enable 'Overwrite existing files' to replace them."
                    })
                if uploaded_success_files:
                    st.session_state.admin_messages.append({
                        "type": "success", 
                        "text": f"✅ Uploaded successfully."
                    })
                
                if new_files_uploaded:
                    with st.spinner("Indexing new documents..."):
                        rebuild_knowledge_base()
                    st.rerun()
                    
    # 2. Uploaded Documents Expander
    with st.sidebar.expander(f"📚 Uploaded Documents ({len(pdf_files)})", expanded=False):
        if pdf_files:
            st.markdown(f"**Total Documents:** `{len(pdf_files)}`")
            for pdf in pdf_files:
                pdf_path = os.path.join(DOCS_DIR, pdf)
                size_str = get_file_size_str(pdf_path)
                st.markdown(f"📄 **{pdf}** (`{size_str}`)")
        else:
            st.info("No documents uploaded yet.")
            
    # 3. Knowledge Base Statistics Expander
    with st.sidebar.expander("📊 Knowledge Base Statistics", expanded=False):
        metadata = load_kb_metadata()
        st.markdown(f"📁 **Total Documents:** `{metadata.get('total_documents', len(pdf_files))}`")
        st.markdown(f"🧩 **Total Chunks:** `{metadata.get('total_chunks', 0)}`")
        st.markdown(f"🤖 **Embedding Model:** `all-MiniLM-L6-v2`")
        st.markdown(f"🗄️ **Vector Database:** `FAISS`")
        st.markdown(f"🕒 **Last Indexed:** `{metadata.get('last_indexed', 'N/A')}`")
        
    # 4. Maintenance Expander
    with st.sidebar.expander("🛠️ Maintenance", expanded=False):
        st.markdown("#### Delete Document")
        if pdf_files:
            doc_to_delete = st.selectbox(
                "Select PDF to delete",
                options=["Select a document..."] + pdf_files,
                key="delete_doc_select"
            )
            if doc_to_delete != "Select a document...":
                if st.button("🗑️ Delete Selected Document", use_container_width=True, type="secondary"):
                    with st.spinner(f"Deleting {doc_to_delete}..."):
                        delete_document(doc_to_delete)
                    st.rerun()
        else:
            st.info("No documents to delete.")
            
        st.markdown("---")
        st.markdown("#### Index Maintenance")
        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True, help="Force rebuild the index"):
            with st.spinner("Rebuilding knowledge base..."):
                rebuild_knowledge_base(force=True)
            st.session_state.admin_messages.append({"type": "success", "text": "✅ Knowledge base rebuilt."})
            st.rerun()
            
    # Debug Mode toggle for admin
    st.sidebar.markdown("---")
    debug_mode = st.sidebar.checkbox("🐞 Debug Mode", value=False)
    
    # Sidebar Debug Information display
    if debug_mode:
        st.sidebar.markdown("### 📄 Knowledge Base Debug Stats")
        st.sidebar.write(f"- Uploaded Documents: `{len(pdf_files)}`")
        st.sidebar.write(f"- Indexed Files: `{', '.join(pdf_files) if pdf_files else 'None'}`")
        
        total_chunks = len(st.session_state.get("chunks", []))
        st.sidebar.write(f"- Total Chunks: `{total_chunks}`")
        
        if st.session_state.index:
            st.sidebar.write(f"- FAISS dimension: `{st.session_state.index.d}`")
            st.sidebar.write(f"- Total vectors stored: `{st.session_state.index.ntotal}`")
            
        st.sidebar.markdown("### 🔍 Last Retrieval Details")
        retrieved_docs = []
        retrieved_chunk_ids = []
        top_hits_list = st.session_state.get("last_query_top_hits", [])
        for hit in top_hits_list[:3]:  # Top 3 retrieved chunks
            meta = hit.get("meta")
            if isinstance(meta, dict):
                retrieved_docs.append(meta.get("filename", "N/A"))
                retrieved_chunk_ids.append(str(meta.get("chunk_id", "N/A")))
            else:
                retrieved_docs.append("Legacy/N/A")
                retrieved_chunk_ids.append(str(hit.get("idx", "N/A")))
                
        st.sidebar.write(f"- Retrieved Docs: `{', '.join(set(retrieved_docs)) if retrieved_docs else 'None'}`")
        st.sidebar.write(f"- Retrieved Chunk IDs: `{', '.join(retrieved_chunk_ids) if retrieved_chunk_ids else 'None'}`")
        
        st.sidebar.markdown("### 🐞 Search Performance Metrics")
        times = st.session_state.get("debug_times", {})
        st.sidebar.write(f"- Embedding/Index Load: `{times.get('embedding', 0.0):.4f}s`")
        st.sidebar.write(f"- FAISS search: `{times.get('faiss_search', 0.0):.4f}s`")
        st.sidebar.write(f"- RAG Routing Audit: `{times.get('rag_routing', 0.0):.4f}s`")
        st.sidebar.write(f"- Groq API: `{times.get('groq_api', 0.0):.4f}s`")
        st.sidebar.write(f"**Last Router Decision**: `{st.session_state.get('last_router_decision', 'N/A')}`")
        st.sidebar.write(f"**Last Similarity Score**: `{st.session_state.get('last_score', 0.0):.4f}`")
        
        with st.sidebar.expander("FAISS Top Search Hits", expanded=False):
            if top_hits_list:
                for hit in top_hits_list:
                    tag = "✅ TOP CHUNK" if hit["rank"] <= 3 else "ℹ️ EXTRA CHUNK"
                    st.caption(f"**Rank {hit['rank']}** (Score: `{hit['score']:.4f}`) - {tag}")
                    meta = hit.get("meta")
                    if isinstance(meta, dict):
                        st.caption(f"Doc: `{meta.get('filename')}` | Chunk: `{meta.get('chunk_id')}` | Page: `{meta.get('page')}`")
                    else:
                        st.caption(f"Chunk ID: `{hit['idx']}`")
                    st.text(hit["text"])
                    st.markdown("---")
            else:
                st.caption("No queries run yet.")
else:
    # Visitor mode: No debug mode
    debug_mode = False

# Main UI Area
st.markdown("<h1 class='title-text'>ContextLens 🔍</h1>", unsafe_allow_html=True)
if pdf_files:
    st.markdown(f"Currently querying: **{len(pdf_files)} document(s)** in Knowledge Base")
else:
    st.markdown("No documents loaded.")

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

# Disable input if no documents are loaded
chat_disabled = (len(pdf_files) == 0 or st.session_state.index is None)
if chat_disabled:
    if mode == "Admin":
        st.warning("⚠️ **Please upload one or more PDFs to the sidebar to start asking questions.**")
    else:
        st.warning("⚠️ **Support Chatbot is currently offline (no documents uploaded). Please contact the administrator.**")

query = st.chat_input("Ask a question about the uploaded documents...", disabled=chat_disabled)

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
        top_hits = []
        
        # Reset query-specific times
        st.session_state.debug_times["faiss_search"] = 0.0
        st.session_state.debug_times["rag_routing"] = 0.0
        st.session_state.debug_times["groq_api"] = 0.0
        
        print("\n=== PIPELINE RUN ===")
        # We will use st.status to show the pipeline steps
        with st.status("🔍 Processing Query through RAG Pipeline...", expanded=True) as status_box:
            try:
                # [STEP 1] Question received
                status_box.write("📥 **[STEP 1] Question received**: " + query)
                print(f"[STEP 1] Question received: {query}")
                
                # [STEP 2] Querying Knowledge Base
                status_box.write(f"📄 **[STEP 2] Querying Knowledge Base** ({len(pdf_files)} documents)")
                print(f"[STEP 2] Querying Knowledge Base: {', '.join(pdf_files)}")
                
                # [STEP 3] FAISS index status
                status_box.write(f"📦 **[STEP 3] FAISS index status**: Ready (contains {len(st.session_state.get('chunks', []))} chunks)")
                print(f"[STEP 3] FAISS index status: Ready")
                
                # [STEP 4] Running similarity search
                status_box.write("🔎 **[STEP 4] Running similarity search** over document chunks...")
                print("[STEP 4] Running similarity search...")
                
                start_t = time.time()
                embedder = get_embedder()
                # Always retrieve top 3 chunks (using threshold = -1.0)
                retrieved_chunks, max_score, top_hits = query_document(
                    query=query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    model=embedder,
                    threshold=-1.0,
                    top_k=3
                )
                st.session_state.debug_times["faiss_search"] = time.time() - start_t
                st.session_state.last_score = max_score
                st.session_state.last_query_top_hits = top_hits
                
                # Print FAISS details to terminal
                print(f"[STEP 4] FAISS total vectors: {st.session_state.index.ntotal if st.session_state.index else 0}")
                print("[STEP 4] Top similarity scores:")
                for hit in top_hits[:5]:
                    print(f"   Rank {hit['rank']}: Score = {hit['score']:.6f}, Chunk Index = {hit['idx']}")
                
                # [STEP 5] Similarity score
                status_box.write(f"📊 **[STEP 5] Similarity score**: {max_score:.4f}")
                print(f"[STEP 5] Similarity score: {max_score:.4f}")
                
                # Debug query_document() to terminal
                print(f"[STEP 5] query_document(): retrieved top {len(retrieved_chunks)} chunks from FAISS")
                for idx, c in enumerate(retrieved_chunks):
                    print(f"   Top Chunk {idx+1} [ID {top_hits[idx]['idx']}] (Score: {top_hits[idx]['score']:.6f}):\n{c[:200]}...")
                
                # Check context sufficiency using Groq
                if retrieved_chunks:
                    context_text = "\n\n".join(retrieved_chunks)
                    
                    status_box.write("🔎 **[STEP 5a] Auditing context sufficiency** using LLM router...")
                    print("[STEP 5a] Auditing context sufficiency using LLM...")
                    
                    router_system_instruction = (
                        "You are an expert context auditor. Your job is to determine whether the provided document context "
                        "contains sufficient information to answer the user's question. Respond with EXACTLY 'YES' or 'NO' "
                        "(no explanation, no punctuation, no extra text, just the word)."
                    )
                    router_prompt = (
                        f"Document Context:\n{context_text}\n\n"
                        f"User Question: {query}\n\n"
                        f"Does the document context contain enough information to answer the user's question? "
                        f"Answer with YES or NO."
                    )
                    
                    start_t = time.time()
                    router_decision = generate_groq_answer(groq_client, router_prompt, router_system_instruction, model_name=groq_model)
                    st.session_state.debug_times["rag_routing"] = time.time() - start_t
                    
                    clean_decision = router_decision.strip().upper()
                    st.session_state.last_router_decision = clean_decision
                    
                    status_box.write(f"📊 **[STEP 5b] LLM Sufficiency Decision**: `{clean_decision}`")
                    print(f"[STEP 5b] LLM Sufficiency Decision: {clean_decision}")
                    
                    if "YES" in clean_decision:
                        source = "📄 Document"
                        
                        # Get unique retrieved document names
                        doc_names = []
                        for hit in top_hits[:len(retrieved_chunks)]:
                            meta = hit.get("meta")
                            if isinstance(meta, dict):
                                doc_names.append(meta.get("filename", "N/A"))
                        unique_docs = []
                        for d in doc_names:
                            if d not in unique_docs:
                                unique_docs.append(d)
                        doc_name = ", ".join(unique_docs) if unique_docs else "N/A"
                        
                        system_instruction = "Answer ONLY using the provided document context. Do not use outside knowledge."
                        prompt = f"Document Context:\n{context_text}\n\nQuestion: {query}"
                        
                        # [STEP 6] Calling Groq
                        status_box.write(f"🤖 **[STEP 6] Calling Groq** using model `{groq_model}`...")
                        print(f"[STEP 6] Calling Groq with model: {groq_model}")
                        
                        start_t = time.time()
                        answer = generate_groq_answer(groq_client, prompt, system_instruction, model_name=groq_model)
                        st.session_state.debug_times["groq_api"] = time.time() - start_t
                        
                        # Append source citation to the answer text
                        citation_text = format_sources(top_hits[:len(retrieved_chunks)])
                        if citation_text:
                            answer += f"\n\n{citation_text}"
                    else:
                        retrieved_chunks = []  # Clear to force fallback below
                
                # Check if we have retrieved chunks/answer
                if not retrieved_chunks:
                    status_box.write("❌ **[STEP 5c] Context insufficient or index empty**. No answer found in document.")
                    print("[STEP 5c] Below threshold / context insufficient, returning fallback response.")
                    answer = "I don't know. I couldn't find that information in the uploaded company documentation."
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
