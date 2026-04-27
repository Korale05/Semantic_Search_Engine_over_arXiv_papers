from fastapi import FastAPI , Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer , CrossEncoder
import numpy as np
import faiss
import json
import os
import time
from pydantic import BaseModel
from typing import List

# ── App setup ─────────────────────────────────────────────

app = FastAPI(
    title="arXiv Sementic Search",
    description="Sementic search over arXiv papers using FAISS + cross-encoder re-ranking",
    version="1.0.0"
)

# Allow frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Load everything at startup ─────────────────────────────
print("Loading papers....")

with open('arxiv_papers.json','r') as file:
    papers = json.load(file)
documents = [f"{p['title']}. {p['abstract']}" for p in papers]

print("Loading models....")

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', local_files_only=True)

print("Loading FAISS index . . .")
index = faiss.read_index('arxiv_hnsw.index')
index.hnsw.efSearch=50

print(f"Ready . {len(papers)} papers indexed")

# ── Response models ────────────────────────────────────────

class PaperResult(BaseModel):
    title : str
    abstract : str
    authors : List[str]
    published : str
    url : str
    rerank_score : float

class SearchResponse(BaseModel):
    query: str
    results: List[PaperResult]
    retrieve_ms: float
    rerank_ms: float
    total_ms: float
    total_papers_indexed: int

# ── Search endpoint ────────────────────────────────────────
@app.get('/search',response_model=SearchResponse)
def search(
    q : str = Query(...,description="Search query",min_length=3),
    top_k : int = Query(5,description="Number of results",ge=1,le=20),
    rerank : bool = Query(True,description="Use cross-encoder re-ranking")
):
    # Stage 1 - FAISS retrivel 
    t1 = time.time()
    query_embedding = bi_encoder.encode([q]).astype('float32')
    candidate_pool = 50 if rerank else top_k
    distances , indices = index.search(query_embedding,candidate_pool)
    retrieve_ms = (time.time() - t1) * 1000

    candidates = []
    for idx in indices[0]:
        p = papers[idx]
        candidates.append({
            'title': p.get('title', 'Unknown Title'),
            'abstract': p.get('abstract', ''),
            'authors': p.get('authors', []),
            'published': p.get('published', 'Unknown'),
            'url': p.get('url', '')
        })
    
    # Stage 2 - Re-ranking
    t2 = time.time()
    if rerank:
        pairs = [[q,f"{c['title']}. {c['abstract'][:300]}"]for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores,candidates),reverse=True)
        top_candidates = [ (score,c) for score , c in ranked[:top_k]]
    else:
        # Use FAISS distances converted to scores
         top_candidates = [
            (1 / (1 + distances[0][i]), candidates[i])
            for i in range(min(top_k, len(candidates)))
        ]
         

    rerank_ms = (time.time() - t2) * 1000
    results = [
        PaperResult(
            title=c['title'],
            abstract=c['abstract'][:500],
            authors=c['authors'],
            published=c['published'],
            url=c['url'],
            rerank_score=round(float(score), 4)
        )
        for score, c in top_candidates
    ]

    return SearchResponse(
        query=q,
        results=results,
        retrieve_ms=round(retrieve_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        total_ms=round(retrieve_ms + rerank_ms, 1),
        total_papers_indexed=len(papers)
    )

# ── Health check endpoint ──────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "papers_indexed": len(papers),
        "models": {
            "bi_encoder": "all-MiniLM-L6-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
    }


# ── Stats endpoint ─────────────────────────────────────────
@app.get("/stats")
def stats():
    return {
        "total_papers": len(papers),
        "embedding_dimension": 384,
        "index_type": "HNSW",
        "evaluation": {
            "bm25_mrr": 0.125,
            "semantic_mrr": 0.900,
            "reranked_mrr": 1.000
        }
    }