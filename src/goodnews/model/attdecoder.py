import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, num_heads=8, dropout=0.1):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions, src_mask, tgt_mask):
        embeddings = self.embed(captions)
        embeddings = self.pos_encoding(embeddings)
        outputs = features.unsqueeze(1) + embeddings
        for layer in self.layers:
            outputs = layer(outputs, src_mask, tgt_mask)
        outputs = self.linear(outputs)
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, src_mask, tgt_mask):
        # Self attention layer
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)[0])
        # Multi-head attention layer
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.multihead_attn(x_norm, x_norm, x_norm, attn_mask=src_mask)[0])
        # Position-wise feedforward layer
        x_norm = self.norm3(x)
        x = x + self.dropout3(F.relu(self.linear2(self.linear1(x_norm))))
        return x
