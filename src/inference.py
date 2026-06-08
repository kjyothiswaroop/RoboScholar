import torch
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from src.model.transformer import Transformer
from src.rag.chunker import Chunker
from src.rag.embedder import Embedder
from src.rag.index import Indexing
from src.rag.retriever import Retriever

SOS_ID = 1; EOS_ID = 2; SEP_ID = 32000

class RoboScholarInference:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = ByteLevelBPETokenizer(
            cfg.paths.tokenizer_vocab,
            cfg.paths.tokenizer_merges
        )

        self.model = Transformer(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            max_len=cfg.model.max_len
        ).to(self.device)
        self.model.load_state_dict(torch.load(cfg.paths.checkpoint, map_location=self.device))
        self.model.eval()

        self.embedder = Embedder()
        self.indexing = Indexing()
        self.retriever = Retriever(self.indexing, self.embedder)

        if Path(cfg.paths.vector_store).exists():
            self.indexing.load(cfg.paths.vector_store)

    def index_pdf(self, pdf_path):
        chunker = Chunker()
        chunks = chunker.chunk_document(pdf_path)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.indexing.add(chunks, embeddings)
        self.indexing.save(self.cfg.paths.vector_store)
        return len(chunks)

    def answer(self, question, max_new_tokens=100):
        retrieved = self.retriever.retrieve(question)
        if not retrieved:
            context = ""
            source = None
        else:
            context = retrieved[0]["text"]
            source = {"source": retrieved[0]["source"], "page": retrieved[0]["page"]}

        context_ids  = self.tokenizer.encode(context).ids
        question_ids = self.tokenizer.encode(question).ids

        max_prefix = self.cfg.model.max_len - max_new_tokens - 3
        context_ids = context_ids[:max_prefix]

        input_ids = [SOS_ID] + context_ids + [SEP_ID] + question_ids + [SEP_ID]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                mask = torch.ones(1, input_tensor.size(1), dtype=torch.long).to(self.device)
                logits, _ = self.model(input_tensor, mask)
                next_token = logits[0, -1, :].argmax(dim=-1).item()
                if next_token == EOS_ID:
                    break
                generated.append(next_token)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)

        answer_text = self.tokenizer.decode(generated)
        return answer_text, source
