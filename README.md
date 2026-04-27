                DOCUMENTS
                    ↓
        ┌───────────┼───────────┐
        ↓                       ↓
     BM25                 Semantic
        ↓                       ↓
 ranked results         ranked results
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
                 MRR
                    ↓
              COMPARISON




        SAME INPUT
            ↓
   ┌────────┴────────┐
   │                 │
 BM25            Semantic
   │                 │
   └──────┬──────────┘
          ↓
        MRR
          ↓
     COMPARISON











                    ┌────────────────────────┐
                    │     DOCUMENTS (Papers) │
                    └────────────┬───────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼

┌───────────────────┐   ┌────────────────────────┐   ┌────────────────────┐
│   BM25 Pipeline   │   │  Semantic Pipeline     │   │  Evaluation Setup  │
└───────────────────┘   └────────────────────────┘   └────────────────────┘

        │                        │                        │
        ▼                        ▼                        ▼

Tokenize Text          Convert to Embeddings       Eval Queries (JSON)
(split words)          (using model)               (query + relevance)

        │                        │                        │
        ▼                        ▼                        ▼

BM25 Index              Vector Index (HNSW)        Relevance Function
(keyword scoring)       (fast vector search)       (keyword match)

        │                        │                        │
        └──────────────┬─────────┴──────────────┬─────────┘
                       │                        │
                       ▼                        ▼

                 USER QUERY (same input)

                       │
        ┌──────────────┼──────────────┐
        │                             │
        ▼                             ▼

 BM25 Search                    Semantic Search
 (keyword match)               (embedding + HNSW)

        │                             │
        ▼                             ▼

 Ranked Results                Ranked Results

        │                             │
        └──────────────┬──────────────┘
                       ▼

              MRR COMPUTATION
      (how early correct result appears)

                       │
                       ▼

         Compare Scores (Final Output)

        ┌──────────────────────────────┐
        │ Semantic MRR vs BM25 MRR     │
        │ e.g. 0.83 vs 0.61            │
        │ → +36% improvement           │
        └──────────────────────────────┘"# Semantic_Search_Engine_over_arXiv_papers" 
