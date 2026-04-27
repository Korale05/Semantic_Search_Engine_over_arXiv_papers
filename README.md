# arXiv Semantic Search Engine

End-to-end semantic search system over research papers.
Same architecture used at Google Search, Meta AI, Amazon.

## Results
| System | MRR |
|--------|-----|
| BM25 (keyword baseline) | 0.40 |
| Semantic Search (FAISS HNSW) | 0.900 |
| Semantic + Cross-encoder Re-ranking | 1.000 |

## Architecture
Query → Bi-encoder (all-MiniLM-L6-v2) → FAISS HNSW retrieval 
→ Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) → Results

## Stack
- Embeddings: sentence-transformers
- Vector index: FAISS HNSW (Meta's production algorithm)
- Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Backend: FastAPI
- Frontend: Streamlit
- Dataset: arXiv papers via arxiv API

## How to run
# Terminal 1
uvicorn api:app --port 8000

# Terminal 2  
streamlit run app.py

## What I learned
- Why semantic search beats keyword search on scientific text
- How bi-encoders vs cross-encoders trade off speed vs accuracy
- FAISS HNSW approximate nearest neighbour indexing
- MRR as an evaluation metric for retrieval systems
- Two-stage retrieve-then-rerank pipeline (industry standard)





