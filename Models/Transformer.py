import torch.nn as nn
import torch

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        x = self.embedding(x)  # 임베딩 레이어에 입력 데이터를 넣어 임베딩된 텐서를 얻습니다.
        return x

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.keys = nn.Linear(vocab_size, vocab_size)
        self.queries = nn.Linear(vocab_size, vocab_size)
        self.values = nn.Linear(vocab_size, vocab_size)
        self.layernorm = nn.LayerNorm(vocab_size)
        self.attention_drop = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(vocab_size, vocab_size)
        

    def forward(self, q, kv=None, mask=None): # y: key/value of encoder, mask: tensor
        residual = q # x: b, vocab_size, 128
        q = self.layernorm(q)
        # split keys, queries and values in num_heads
        queries = self.queries(q)
        queries = queries.view(queries.shape[0], self.num_heads, queries.shape[1], -1) # b, vocab_size, 128 -> b, 8(head), vocab_size, 16
        if kv is None: # self attention
            keys = self.keys(q)
            values = self.values(q)
        else: # encoder-decoder attention
            keys = self.keys(kv)
            values = self.values(kv)
        keys = keys.view(keys.shape[0], self.num_heads, keys.shape[1], -1) # b, vocab_size, 128 -> b, 8(head), vocab_size, 16
        values = values.view(values.shape[0], self.num_heads, values.shape[1], -1) # b, vocab_size, 128 -> b, 8(head), vocab_size, 16
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # Q*K^T, energy: b, 8, vocab_size, vocab_size
        scaling = self.vocab_size ** (1/2)
        attention = torch.softmax(energy, dim=-1) / scaling
        if mask is not None: # masked self attention
            attention = attention.masked_fill_(mask == 0, float('-inf'))
        attention = self.attention_drop(attention)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', attention, values) # att*V, out: b, 8, vocab_size, 16
        out = out.view(out.shape[0], out.shape[2], out.shape[1]*out.shape[3]) # b, 8, vocab_size, 16 -> b, vocab_size, 128
        out = self.projection(out)
        out = out + residual
        return out

class FeedForwardNetwork(nn.Sequential):
    def __init__(self, vocab_size, expansion=8, dropout_rate=0.1): # basic model ffn dimension = 2048
        super(FeedForwardNetwork, self).__init__()
        self.layernorm = nn.LayerNorm(vocab_size)
        self.linear1 = nn.Linear(vocab_size, expansion * vocab_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(expansion * vocab_size, vocab_size)
    def forward(self, x):
        residual = x # x: b, vocab_size, 128
        x = self.layernorm(x)
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual
        return out

# Define the transformer
class Transformer(nn.Sequential):
    def __init__(self, vocab_size, d_model=512, dropout_rate=0.1, num_heads=8, expansion=8, depth=8):
        super(Transformer, self).__init__()
        self.ie = InputEmbedding(vocab_size, d_model=d_model)
        self.mha = MultiHeadAttention(vocab_size, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn = FeedForwardNetwork(vocab_size, expansion=expansion, dropout_rate=dropout_rate)
        self.layer_norm = nn.LayerNorm(vocab_size)
        self.linear = nn.Linear(vocab_size, vocab_size)
        self.vocab_size = vocab_size
        self.depth = depth
    
    def forward(self, x, y): # x: encoder input, y: decoder input
        x = self.ie(x)
        y = self.ie(y)
        for i in range(self.depth):
            x = self.mha(q=x)
            x = self.ffn(x)
        Mask = torch.tril(torch.ones(self.vocab_size, self.vocab_size))
        for i in range(self.depth):
            y = self.mha(q=y, Mask=Mask) # masked self-head attention
            y = self.mha(q=y, kv=x) # encoder-decoder attention
            y = self.ffn(y)
        # Linear layer for classification
        output = self.linear(y) # y: b, vocab_size, 128 -> b, vocab_size, 128
        return output