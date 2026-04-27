# app.py
import streamlit as st
import requests
import time

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="arXiv Semantic Search",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ── Header ────────────────────────────────────────────────
st.title("🔍 arXiv Semantic Search")
st.markdown("Search over **21,536 research papers** using semantic embeddings + cross-encoder re-ranking")

# ── Sidebar — stats ───────────────────────────────────────
with st.sidebar:
    st.header("📊 System Info")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        st.metric("Papers Indexed", stats["total_papers"])
        st.metric("Embedding Dim", stats["embedding_dimension"])
        st.metric("Index Type", stats["index_type"])
        
        st.divider()
        st.subheader("Evaluation Results")
        eval_data = stats["evaluation"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BM25 MRR", eval_data["bm25_mrr"], 
                     delta=None, help="Keyword search baseline")
        with col2:
            st.metric("Semantic MRR", eval_data["semantic_mrr"],
                     delta=f"+{((eval_data['semantic_mrr']-eval_data['bm25_mrr'])/eval_data['bm25_mrr']*100):.0f}%")
        
        st.metric("Re-ranked MRR", eval_data["reranked_mrr"],
                 delta="Perfect score")
        
    except:
        st.error("API not reachable. Is uvicorn running?")
    
    st.divider()
    st.subheader("⚙️ Search Settings")
    top_k = st.slider("Number of results", 1, 10, 5)
    use_rerank = st.toggle("Cross-encoder re-ranking", value=True)
    if use_rerank:
        st.caption("✅ More accurate, ~2s on CPU")
    else:
        st.caption("⚡ Faster, FAISS only")

# ── Search bar ────────────────────────────────────────────
st.divider()
query = st.text_input(
    "Enter your search query",
    placeholder="e.g. attention mechanism in transformers",
    label_visibility="collapsed"
)

example_queries = [
    "BERT pre-training language model",
    "reinforcement learning policy gradient",
    "image classification neural network",
    "text summarization abstractive"
]

st.caption("Try: " + " · ".join(
    f"`{q}`" for q in example_queries
))

# ── Search execution ──────────────────────────────────────
if query and len(query.strip()) >= 3:
    with st.spinner("Searching..."):
        try:
            response = requests.get(
                f"{API_URL}/search",
                params={"q": query, "top_k": top_k, "rerank": use_rerank},
                timeout=30
            )
            data = response.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # Timing info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retrieve", f"{data['retrieve_ms']:.0f}ms")
    with col2:
        st.metric("Re-rank", f"{data['rerank_ms']:.0f}ms")
    with col3:
        st.metric("Total", f"{data['total_ms']:.0f}ms")

    st.divider()
    st.subheader(f"Results for: *{query}*")

    # Result cards
    for i, paper in enumerate(data["results"]):
        with st.container():
            col_num, col_content = st.columns([0.05, 0.95])
            
            with col_num:
                st.markdown(f"### {i+1}")
            
            with col_content:
                # Title as clickable link
                st.markdown(f"### [{paper['title']}]({paper['url']})")
                
                # Authors and date
                authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
                st.caption(f"👤 {authors_str} · 📅 {paper['published']} · Score: {paper['rerank_score']:.3f}")
                
                # Abstract
                with st.expander("Abstract"):
                    st.write(paper["abstract"])
            
            st.divider()

elif query and len(query.strip()) < 3:
    st.warning("Query must be at least 3 characters.")