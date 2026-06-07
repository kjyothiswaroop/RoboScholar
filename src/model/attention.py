import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0 , "d_model must be divisible by number of heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        self.W_output = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        Q = self.W_query(query).view(batch, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_key(key).view(batch, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_value(value).view(batch, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(batch, seq_len_q, self.d_model)
        return self.W_output(out), weights