import torch
import torch.nn as nn
from src.model.embeddings import Embeddings
from src.model.transformer_block import TransformerBlock

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1, max_len=512):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, dropout, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):

        x = self.embeddings(x)
        all_weights = []
        for block in self.blocks:
            x, weights = block(x, mask)
            all_weights.append(weights)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits, all_weights