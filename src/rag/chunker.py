import fitz

class Chunker:

    def __init__(self, chunk_size = 256, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, path):
        chunks = []
        source = path.split("/")[-1]
        with fitz.open(path) as f:
            for page_num, page in enumerate(f):
                text = page.get_text()
                chunks.extend(self._split_into_chunks(text, page_num + 1, source))
        return chunks
    
    def _split_into_chunks(self, text, page, source):
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "source": source,
                "page": page,
                "chunk_id": f"{source}_p{page}_i{i}"
                
            })
            
        return chunks