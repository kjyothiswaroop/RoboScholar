import arxiv
from tqdm import tqdm
import os
import json

class DataExporter:

    def __init__(self, query, max_results, output_path):

        self.query = query
        self.max_results = max_results
        self.output_path = output_path

    def fetch(self):

        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []

        for result in tqdm(search.results(), total=self.max_results):
            papers.append({
                "id" : result.entry_id,
                "title" : result.title,
                "abstract" : result.summary,
                "authors" : [a.name for a in result.authors],
                "published" : str(result.published)
            })

        return papers

    def export(self):
        
        existing_ids = set()
        
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                for line in f:
                    existing_ids.add(json.loads(line)["id"])

        papers = self.fetch()

        with open(self.output_path, "a") as f:
            for paper in papers:
                if paper["id"] not in existing_ids:
                    f.write(json.dumps(paper) + "\n")
    
if __name__ == "__main__":
    exporter = DataExporter(
        query="cat:cs.RO",
        max_results=10000,
        output_path="../../data/raw/abstracts.jsonl"
    )
    
