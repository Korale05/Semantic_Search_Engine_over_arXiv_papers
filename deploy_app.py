import streamlit as st
import numpy as np
import faiss
import json
import os
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="arXiv Semantic Search",
    page_icon="🔍",
    layout="wide"
)

# ── Your HuggingFace repo ──────────────────────────────────
HF_REPO_ID = "onkar1718/arxiv-semantic-search"  # ← change this

# ── Load everything (cached so it only runs once) ─────────
@st.cache_resource
def load_system():
    
    # --- Download FAISS index from HuggingFace if not present locally ---
    if not os.path.exists("arxiv_hnsw.index"):
        with st.spinner("📥 Downloading FAISS index from HuggingFace..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="arxiv_hnsw.index",
                repo_type="model",
                local_dir="."          # saves into current folder
            )
    
    # --- Download papers JSON from HuggingFace if not present locally ---
    if not os.path.exists("arxiv_papers.json"):
        with st.spinner("📥 Downloading papers data from HuggingFace..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="arxiv_papers.json",
                repo_type="model",
                local_dir="."
            )

    # --- Load papers ---
    with open('arxiv_papers.json', 'r') as f:
        papers = json.load(f)

    documents = [f"{p['title']}. {p['abstract']}" for p in papers]

    # --- Load models (these auto-download from HuggingFace, no change needed) ---
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # --- Load FAISS index ---
    index = faiss.read_index('arxiv_hnsw.index')
    index.hnsw.efSearch = 50

    return papers, documents, bi_encoder, cross_encoder, index

papers, documents, bi_encoder, cross_encoder, index = load_system()

# ── rest of your code stays exactly the same ──────────────

# ── Search functions ──────────────────────────────────────
def search_semantic(query, top_k=5):
    query_emb = bi_encoder.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        p = papers[idx]
        results.append({**p, 'distance': float(dist)})
    return results

def search_with_reranking(query, top_k=5, candidate_pool=50):
    t1 = time.time()
    query_emb = bi_encoder.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, candidate_pool)
    retrieve_ms = (time.time() - t1) * 1000

    candidates = [papers[idx] for idx in indices[0]]

    t2 = time.time()
    pairs = [[query, f"{c['title']}. {c['abstract'][:300]}"] for c in candidates]
    scores = cross_encoder.predict(pairs).tolist()
    rerank_ms = (time.time() - t2) * 1000

    ranked = sorted(zip(scores, candidates), reverse=True)
    top_results = [c for _, c in ranked[:top_k]]
    top_scores = [float(s) for s, _ in ranked[:top_k]]

    return top_results, top_scores, retrieve_ms, rerank_ms

# ── UI ────────────────────────────────────────────────────
st.title("🔍 arXiv Semantic Search")
st.markdown(f"Search over **{len(papers):,} research papers** using semantic embeddings + cross-encoder re-ranking")

with st.sidebar:
    st.header("📊 Evaluation Results")
    st.metric("BM25 MRR", "0.708", help="Keyword search baseline")
    st.metric("Semantic MRR", "0.512")
    st.metric("Semantic MRR + Re-ranked MRR", "0.875", delta="Perfect score")
    st.metric("Hybrid RRF", "0.646")
    st.metric("Hybrid + Re-ranking ", "0.850")



    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results", 1, 10, 5)
    use_rerank = st.toggle("Cross-encoder re-ranking", value=True)
    if use_rerank:
        st.caption("✅ More accurate, ~1–2s")
    else:
        st.caption("⚡ Faster, FAISS only")

    st.divider()
    st.header("🏗️ Stack")
    st.caption("Bi-encoder: all-MiniLM-L6-v2")
    st.caption("Index: FAISS HNSW")
    st.caption("Re-ranker: ms-marco-MiniLM-L-6-v2")
    st.caption("21,536 arXiv papers")

st.divider()
query = st.text_input(
    "Search query",
    placeholder="e.g. attention mechanism in transformers",
    label_visibility="collapsed"
)

st.caption("Try: `BERT pre-training` · `reinforcement learning policy gradient` · `image segmentation CNN` · `GAN image synthesis`")

if query and len(query.strip()) >= 3:
    with st.spinner("Searching..."):
        if use_rerank:
            results, scores, retrieve_ms, rerank_ms = search_with_reranking(query, top_k)
            total_ms = retrieve_ms + rerank_ms
        else:
            t = time.time()
            raw = search_semantic(query, top_k)
            retrieve_ms = (time.time() - t) * 1000
            results = raw
            scores = [1/(1+r['distance']) for r in raw]
            rerank_ms = 0
            total_ms = retrieve_ms

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieve", f"{retrieve_ms:.0f}ms")
    col2.metric("Re-rank", f"{rerank_ms:.0f}ms")
    col3.metric("Total", f"{total_ms:.0f}ms")

    st.divider()
    st.subheader(f"Results for: *{query}*")

    for i, (paper, score) in enumerate(zip(results, scores)):
        col_n, col_c = st.columns([0.05, 0.95])
        with col_n:
            st.markdown(f"### {i+1}")
        with col_c:
            st.markdown(f"### [{paper['title']}]({paper['url']})")
            authors = ", ".join(paper.get('authors', [])) or "Unknown"
            st.caption(f"👤 {authors} · 📅 {paper.get('published','?')} · Score: {score:.3f}")
            with st.expander("Abstract"):
                st.write(paper.get('abstract', ''))
        st.divider()