import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # inherit parent class nn.Module
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (heads * self.head_dim == embed_size), "embed_size must be dividible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * heads, self.embed_size, bias=False)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embed_size into heads of size head_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Compute V, Q, K
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, head_dim)

        # Compute Q*K.t
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim) ** (1 / 2), dim=3)  # (N, heads, query_len, key_len)

        # dim=3 normalize across keys, so that the values in the column of each matrix query_len*key_len
        # sum up to 1

        # Attention times V and also flatten the last two dimensions from (N, query_len, heads, head_dim) to (N,
        # query_len, embed_size)
        out = torch.einsum("nhql , nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # multiply by WO weight matrix
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
