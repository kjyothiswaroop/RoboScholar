import arxivscraper
import fitz
import requests
import time
from tqdm import tqdm
from pathlib import Path
import os
import json


class DataExporter:

    def __init__(self, category, date_from, date_until, output_path, max_papers=5000):
        self.category = category
        self.date_from = date_from
        self.date_until = date_until
        self.output_path = Path(output_path)
        self.max_papers = max_papers

    def fetch(self):
        scraper = arxivscraper.Scraper(
            category="cs.RO",
            date_from=self.date_from,
            date_until=self.date_until,
        )
        papers = scraper.scrape()
        for p in papers:
            yield {
                "id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "authors": p["authors"],
                "published": p["created"],
            }

    def parse_pdf(self, paper_id):
        url = f"https://arxiv.org/pdf/{paper_id}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            doc = fitz.open(stream=response.content, filetype="pdf")
            full_text = "\n".join(page.get_text() for page in doc)
            return full_text.strip()
        except Exception:
            return None

    def export(self):
        existing_ids = set()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                for line in f:
                    existing_ids.add(json.loads(line)["id"])

        added = 0
        with open(self.output_path, "a") as f:
            for paper in tqdm(self.fetch(), total=self.max_papers):
                if added >= self.max_papers:
                    break
                if paper["id"] in existing_ids:
                    continue

                full_text = self.parse_pdf(paper["id"])
                paper["full_text"] = full_text if full_text else paper["abstract"]

                f.write(json.dumps(paper) + "\n")
                f.flush()
                existing_ids.add(paper["id"])
                added += 1
                time.sleep(3)

        print(f"Saved {added} papers to {self.output_path}")
        return added


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent.parent
    exporter = DataExporter(
        category="cs",
        date_from="2024-01-01",
        date_until="2026-05-30",
        output_path=ROOT / "data" / "raw" / "papers.jsonl",
        max_papers=5000
    )
    exporter.export()
