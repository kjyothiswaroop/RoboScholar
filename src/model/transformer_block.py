import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention
class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        if mask is not None:
            causal_mask = causal_mask * mask.unsqueeze(1)

        attention_output, weights = self.attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attention_output))

        x = self.norm2(x + self.dropout(self.ff(x)))

        return x , weights