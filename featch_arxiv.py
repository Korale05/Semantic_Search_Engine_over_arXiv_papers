
import arxiv
import json
import time
import os

def fetch_papers(query,max_results=500):
    print(f"Featching '{query}' papers from arXiv...")
    
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in client.results(search):
        papers.append({
            'id' : result.entry_id,
            'title':result.title,
            'abstract' : result.summary,
            'authors':[a.name for a in result.authors[:3]],
            'published' : str(result.published.date()),
            'url' : result.entry_id
        })

        if len(papers) % 100 ==0:
            print(f" Featched {len(papers)} papers...")
    
    return papers

# Featch from multiple topics to get diversity

topics = [
    "machine learning",
    "natural language processing", 
    "computer vision",
    "deep learning optimization"
]

all_papers = []

for topic in topics:
    papers = fetch_papers(topic,max_results=500)
    all_papers.extend(papers)
    print(f"Topics '{topic}' : {len(papers)} papers featched")
    time.sleep(5) # Wait between topics

# Remove duplicated by paper ID
seen_ids = set()
unique_papers = []
for p in all_papers:
    if p['id'] not in seen_ids:
        seen_ids.add(p['id'])
        unique_papers.append(p)

print(f"Total unique papers : {len(unique_papers)}")

# Save to disk
with open('arxiv_papers.json','w') as file:
    json.dump(unique_papers,file,indent=2)

print("Saved to arxiv_papers.json")
