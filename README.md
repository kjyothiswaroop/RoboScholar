# RoboScholar

A from-scratch encoder-decoder transformer for robotics paper Q&A, with RAG, explainability, and an MCP server interface.

## Setup

```bash
uv sync
ollama pull llama3.1:8b
```

## Data Generation

### 1. Scrape Papers

Fetches full robotics papers (cs.RO) from arXiv using the OAI-PMH bulk API. Downloads and parses full PDFs using PyMuPDF. Saves to `data/raw/papers.jsonl`.

```bash
python scripts/data/arxiv_scrapper.py
```

### 2. Generate Q&A Pairs

Uses `llama3.1:8b` via Ollama to generate 3 question-answer pairs per paper. Each triple contains a `question`, `answer`, and `relevant_excerpt` (a direct quote from the paper that supports the answer). The excerpt serves as the encoder context during training. Saves to `data/qa_pairs/qa_pairs.jsonl`.

```bash
python scripts/data/qa_generator.py
```

> Run both scripts in a tmux session on a remote machine for overnight execution. Resume support is built in — both scripts skip already processed entries on restart.
