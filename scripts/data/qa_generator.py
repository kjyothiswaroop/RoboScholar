import ollama
import json
from pathlib import Path
from tqdm import tqdm

class QAGenerator:

    def __init__(self, input_path, output_path, model):
        self.in_path = input_path
        self.out_path = output_path
        self.model = model
        
    
    def generate_qa(self, paper):

        prompt = f"""You are a scientific Q&A generator for robotics papers.

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
            ]
        """

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        text = response.message.content

        try:
            pairs = json.loads(text)
            return pairs

        except json.JSONDecodeError:
            return []
        
    
    def run(self):
        existing_ids = set()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.out_path.exists():
            with open(self.out_path, "r") as f:
                for line in f:
                    existing_ids.add(json.loads(line)["paper_id"])
        
        with open(self.in_path, "r") as infile, open(self.out_path, "a") as outfile:
            papers = [json.loads(line) for line in infile]
            for paper in tqdm(papers):
                if(paper["id"]) in existing_ids:
                    continue
                
                pairs = self.generate_qa(paper)
                for pair in pairs:
                    pair["paper_id"] = paper["id"]
                    outfile.write(json.dumps(pair) + "\n")
                    outfile.flush()

if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent.parent
    generator = QAGenerator(
        input_path=ROOT / "data" / "raw" / "papers.jsonl",
        output_path=ROOT / "data" / "qa_pairs" / "qa_pairs.jsonl",
        model="llama3.1:8b"
    )
    generator.run()