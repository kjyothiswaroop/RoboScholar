import faiss
import numpy as np
import pickle
from pathlib import Path

class Indexing:

    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.chunks = []

    def add(self, chunks, embeddings):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
