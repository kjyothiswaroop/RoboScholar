
class Retriever:

    def __init__(self, indexing, embedder, k=3):
        self.indexer = indexing
        self.embedder = embedder
        self.k = k
    
    def retrieve(self, question, source=None):
        query_vector = self.embedder.embed([question]).astype("float32")

        # if filtering by source, search the whole index then keep only matching
        # chunks; otherwise just grab the global top-k
        search_k = self.indexer.index.ntotal if source else self.k
        if search_k == 0:
            return []

        distances, indices = self.indexer.index.search(query_vector, search_k)

        results = []
        for i in indices[0]:
            if i < 0 or i >= len(self.indexer.chunks):
                continue
            chunk = self.indexer.chunks[i]
            if source and chunk["source"] != source:
                continue
            results.append(chunk)
            if len(results) == self.k:
                break
        return results