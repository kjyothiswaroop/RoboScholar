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

        if Path(f"{cfg.paths.vector_store}/index.faiss").exists():
            self.indexing.load(cfg.paths.vector_store)

    def index_pdf(self, pdf_path):
        filename = pdf_path.split("/")[-1]
        if any(c["source"] == filename for c in self.indexing.chunks):
            return 0

        chunker = Chunker()
        chunks = chunker.chunk_document(pdf_path)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.indexing.add(chunks, embeddings)
        self.indexing.save(self.cfg.paths.vector_store)
        return len(chunks)

    def list_papers(self):
        return sorted({c["source"] for c in self.indexing.chunks})

    def answer(self, question, source=None, max_new_tokens=100, temperature=0.8, top_k=40, repetition_penalty=1.3):
        retrieved = self.retriever.retrieve(question, source=source)
        if not retrieved:
            context = ""
            source = None
            excerpt = ""
        else:
            context = retrieved[0]["text"]
            source = {"source": retrieved[0]["source"], "page": retrieved[0]["page"]}
            excerpt = retrieved[0]["text"]

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
                next_logits = logits[0, -1, :]

                # repetition penalty: push down tokens already generated
                for tok in set(generated):
                    if next_logits[tok] > 0:
                        next_logits[tok] /= repetition_penalty
                    else:
                        next_logits[tok] *= repetition_penalty

                # temperature: flatten/sharpen the distribution
                next_logits = next_logits / temperature

                # top-k: keep only the k most likely tokens, then sample
                top_values, top_indices = torch.topk(next_logits, top_k)
                probs = torch.softmax(top_values, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = top_indices[sampled].item()

                if next_token == EOS_ID:
                    break
                generated.append(next_token)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)

        answer_text = self.tokenizer.decode(generated)
        return answer_text, source, excerpt
