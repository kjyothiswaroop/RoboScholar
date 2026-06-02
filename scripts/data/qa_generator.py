import re
import requests
import json
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SGLANG_URL = "http://129.105.69.10:3000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


class QAGenerator:

    def __init__(self, input_path, output_path, workers=8):
        self.in_path = Path(input_path)
        self.out_path = Path(output_path)
        self.failed_path = self.out_path.with_name("failed_ids.txt")
        self.workers = workers
        self._lock = threading.Lock()

    def _build_prompt(self, paper):
        return f"""You are a scientific Q&A generator for robotics papers.

Given the following robotics paper, generate 3 question-answer pairs.
The questions should be specific to this paper's contributions, methods, or results.
For each pair, also extract a short relevant excerpt from the text that directly supports the answer.

Title: {paper['title']}
Text: {paper['full_text']}

Rules:
- Questions must be answerable from the text only
- Answers should be concise (1-3 sentences)
- relevant_excerpt must be a direct quote from the text
- Return ONLY valid JSON, no explanation, no markdown

[
  {{"question": "...", "answer": "...", "relevant_excerpt": "..."}},
  {{"question": "...", "answer": "...", "relevant_excerpt": "..."}},
  {{"question": "...", "answer": "...", "relevant_excerpt": "..."}}
]"""

    def _parse_response(self, text):
        # Strip markdown fences if the model wrapped the JSON
        match = _FENCE_RE.search(text)
        if match:
            text = match.group(1)
        return json.loads(text.strip())

    def generate_qa(self, paper):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": self._build_prompt(paper)}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        try:
            response = requests.post(SGLANG_URL, json=payload, timeout=120)
            text = response.json()["choices"][0]["message"]["content"]
            return self._parse_response(text), None
        except Exception as e:
            return [], str(e)

    def run(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        successful_ids = set()
        if self.out_path.exists():
            with open(self.out_path) as f:
                for line in f:
                    successful_ids.add(json.loads(line)["paper_id"])

        failed_ids = set()
        if self.failed_path.exists():
            with open(self.failed_path) as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        failed_ids.add(pid)

        already_attempted = successful_ids | failed_ids

        with open(self.in_path) as f:
            papers = [json.loads(line) for line in f]

        papers_to_process = [p for p in papers if p["id"] not in already_attempted]
        print(
            f"Total: {len(papers)} | Done: {len(successful_ids)} | "
            f"Failed: {len(failed_ids)} | To process: {len(papers_to_process)}"
        )

        with open(self.out_path, "a") as outfile, open(self.failed_path, "a") as failfile:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.generate_qa, paper): paper
                           for paper in papers_to_process}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    paper = futures[future]
                    pairs, err = future.result()
                    with self._lock:
                        if pairs:
                            for pair in pairs:
                                pair["paper_id"] = paper["id"]
                                outfile.write(json.dumps(pair) + "\n")
                            outfile.flush()
                        else:
                            failfile.write(paper["id"] + "\n")
                            failfile.flush()


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent.parent
    generator = QAGenerator(
        input_path=ROOT / "data" / "raw" / "papers.jsonl",
        output_path=ROOT / "data" / "qa_pairs" / "qa_pairs.jsonl",
        workers=8
    )
    generator.run()
