import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        depths = torch.arange(0, d_model, 2).float() / d_model
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle_rads)
        pe[:, 1::2] = torch.cos(angle_rads)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Embeddings(nn.Module):

    def __init__(self, vocab_size, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.pos_enc(self.token_emb(x)))