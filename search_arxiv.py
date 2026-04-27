"""
User Query
     ↓
Search System (BM25 or Semantic)
     ↓
Top Results
     ↓
Check: Is correct result early?
     ↓
Score (MRR)

"""



from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import time

# -----Load Papers----------------------------------------------------------
with open('arxiv_papers.json','r') as file:
    papers = json.load(file)

print(f"Loaded {len(papers)} papers.....")

# What we embed = title + abstract together 
# Title carries the topic, abstract carries the detail

documents = [f"{p['title']}. {p['abstract']} "for p in papers]

# ── Embeddings ────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_FILE = 'arxiv_embeddings.npy'
INDEX_FILE = 'arxiv_hnsw.index'

if os.path.exists(EMBEDDING_FILE):
    print("Loading saved embeddings...")
    embeddings = np.load(EMBEDDING_FILE)
else :
    print("Computing embeddings")
    embeddings = model.encode(
        documents,
        show_progress_bar = True,
        batch_size=64
    )
    embeddings = embeddings.astype('float32')
    np.save(EMBEDDING_FILE,embeddings)
    print("Embeddings saved.")

print(f"Embeddings shpae : {embeddings.shape}")


# ── FAISS HNSW Index ──────────────────────────────────────

if os.path.exists(INDEX_FILE):
    print("Loading saved HNSW index ...")
    index = faiss.read_index(INDEX_FILE)
else :
    print("Building HNSW index ....")
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension,32) # (dimension,M) M- no of connection in graph
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    faiss.write_index(index,INDEX_FILE)
    print(f"Index build. Total vectors : {index.ntotal}")


index.hnsw.efSearch = 50

# ── Search function ───────────────────────────────────────

def search(query,top_k=5):
    query_embedding = model.encode([query]).astype('float32')
    distances , indices = index.search(query_embedding,top_k)

    results = []
    for dist , idx in zip(distances[0] ,indices[0]):
        paper = papers[idx]
        results.append({
            'title' : paper['title'],
            'abstract' : paper['abstract'][:300],
            'authors' : paper['authors'],
            'published' : paper['published'],
            'url' : paper['url'],
            'distance' : dist
        })
    return results

# ── Test ──────────────────────────────────────────────────

queries = [
    "transformers for text classification",
    "reinforcement learning reward optimization",
    "image segmentation convolutional networks"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)
    
    start = time.time()
    results = search(query, top_k=3)
    elapsed = (time.time() - start) * 1000
    
    print(f"Search time: {elapsed:.2f}ms\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['title']}")
        print(f"     Authors: {', '.join(r['authors'])}")
        print(f"     Published: {r['published']}")
        print(f"     Distance: {r['distance']:.3f}")
        print(f"     {r['abstract']}...")
        print()



# ── BM25 Search ──────────────────────────────────────────────────

from rank_bm25 import BM25Okapi

# Build BM25 index
print("Building BM25 index ...")

# BM25 needs tokenized text - just split by space
tokenized_docs = [doc.lower() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
print("BM25 index ready.")

def search_bm25(query,top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        paper = papers[idx]
        results.append({
            'title' : paper['title'],
            'abstract' : paper['abstract'][:300],
            'score' : scores[idx]
        })
    return results 




# ── MRR evaluation ──────────────────────────────────────────────────

import json

# Load eval queries
with open('eval_queries.json', 'r') as f:
    eval_queries = json.load(f)

def is_relevant(result, relevant_keywords):
    """Check if a result is relevant based on keywords in title/abstract"""
    text = (result['title'] + ' ' + result['abstract']).lower()
    # Relevant if ANY keyword matches
    return any(kw.lower() in text for kw in relevant_keywords)

def compute_mrr(search_fn, eval_queries, top_k=10):
    """
    MRR = Mean Reciprocal Rank
    For each query, find position of first relevant result.
    Score = 1/position. Average across all queries.
    Perfect score = 1.0 (relevant result always #1)
    """
    reciprocal_ranks = []
    
    for item in eval_queries:
        query = item['query']
        relevant_keywords = item['relevant_keywords']
        
        results = search_fn(query, top_k=top_k)
        
        # Find rank of first relevant result
        rr = 0.0
        for rank, result in enumerate(results, start=1):
            if is_relevant(result, relevant_keywords):
                rr = 1.0 / rank
                break  # Only care about first relevant result
        
        reciprocal_ranks.append(rr)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Run evaluation
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

semantic_mrr = compute_mrr(search, eval_queries)
bm25_mrr = compute_mrr(search_bm25, eval_queries)

print(f"Semantic Search MRR : {semantic_mrr:.3f}")
print(f"BM25 Keyword MRR    : {bm25_mrr:.3f}")

if semantic_mrr > bm25_mrr:
    improvement = ((semantic_mrr - bm25_mrr) / bm25_mrr) * 100
    print(f"\nSemantic search is better by {improvement:.1f}%")
else:
    improvement = ((bm25_mrr - semantic_mrr) / semantic_mrr) * 100
    print(f"\nBM25 is better by {improvement:.1f}%")
    print("(This means you need more/better data — not that your system is broken)")




# ── Cross Encoder (Re-Ranking) ──────────────────────────────────────────────────



"""
So the production pattern is always:

FAISS retrieves top 50 candidates (fast, approximate)
         ↓
Cross-encoder re-ranks top 50 → returns top 5 (slow, precise)
         ↓
User sees top 5 results

"""
from sentence_transformers import CrossEncoder
import time

# Load cross-encoder model
# This model takes (query, document) pairs and scores them
print("Loading cross-encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-encoder ready .")

def search_with_reranking(query,top_k=5,candidate_pool = 50):
    """
    Stage 1: FAISS retrieves top 50 candidates fast
    Stage 2: Cross-encoder re-ranks them precisely
    Stage 3: Return top_k results

    """
    # Stage 1 - retrive candidates
    t1 = time.time()
    query_embedding = model.encode([query]).astype('float32')
    dist , indices = index.search(query_embedding,candidate_pool)
    retrieve_time = (time.time() - t1) * 1000

    # Stage 2 - re-rank with cross encoder
    t2 = time.time()
    candidates = []
    for idx in indices[0]:
        paper = papers[idx]
        candidates.append({
            'title' : paper['title'],
            'abstract' : paper['abstract'],
            'authors' : paper['authors'],
            'published' : paper['published'],
            'url' : paper['url'],
            'index' : idx
        })
    
    # Cross-encoder scores each (query, title+abstract) pair together

    pairs = [ [query,f"{c['title']} . {c['abstract'][:300]}"] for c in candidates]
    scores = cross_encoder.predict(pairs)
    rerank_time = (time.time() - t2)*1000

    # Sort by cross-encoder score (higher = better)
    ranked = sorted(zip(scores,candidates),reverse=True)
    top_results = [c for _, c in ranked[:top_k]]

    print(f"  Retrieve: {retrieve_time:.1f}ms | Re-rank: {rerank_time:.1f}ms | "
          f"Total: {retrieve_time+rerank_time:.1f}ms")
    
    return top_results


# Compare FAISS-only vs FAISS + re-ranking

test_query = "attention mechanism transformer self-attention"

print(f"\n{'='*60}")
print(f"Query: '{test_query}'")
print(f"{'='*60}")

print("\n--- FAISS only (top 5) ---")
for i, r in enumerate(search(test_query, top_k=5)):
    print(f"[{i+1}] {r['title']}")
    print(f"     Distance: {r['distance']:.3f}")

print("\n--- FAISS + Cross-encoder re-ranking (top 5 from 50) ---")
for i, r in enumerate(search_with_reranking(test_query, top_k=5, candidate_pool=50)):
    print(f"[{i+1}] {r['title']}")


def search_reranked_for_eval(query,top_k=10):
    """Wrapper so compute_mrr can use re-ranked search"""
    results = search_with_reranking(query,top_k)
    # Add dummy distance field so is_relevant works
    for r in results:
        r['abstract'] = r['abstract'][:300]
    return results


reranked_mrr = compute_mrr(search_reranked_for_eval, eval_queries)

print(f"\nSemantic Search MRR       : {semantic_mrr:.3f}")
print(f"BM25 Keyword MRR          : {bm25_mrr:.3f}")
print(f"Semantic + Re-ranking MRR : {reranked_mrr:.3f}")






