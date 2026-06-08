
class Retriever:

    def __init__(self, indexing, embedder, k=3):
        self.indexer = indexing
        self.embedder = embedder
        self.k = k
    
    def retrieve(self, question):
        query_vector = self.embedder.embed([question]).astype("float32")
        distances , indices = self.indexer.index.search(query_vector, self.k)
        return [self.indexer.chunks[i] for i in indices[0] if i < len(self.indexer.chunks)]