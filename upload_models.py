# upload_models.py  ← run this only once locally!
from huggingface_hub import HfApi

api = HfApi()

REPO_ID = "onkar1718/arxiv-semantic-search"  # ← change this

# Upload FAISS index
api.upload_file(
    path_or_fileobj="arxiv_hnsw.index",
    path_in_repo="arxiv_hnsw.index",
    repo_id=REPO_ID,
    repo_type="model"
)
print("✅ FAISS index uploaded!")

# Upload papers JSON
api.upload_file(
    path_or_fileobj="arxiv_papers.json",
    path_in_repo="arxiv_papers.json",
    repo_id=REPO_ID,
    repo_type="model"
)
print("✅ Papers JSON uploaded!")