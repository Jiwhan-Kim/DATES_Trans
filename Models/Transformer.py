import torch.nn as nn
import torch

class InputEmbedding(nn.Module):
    def __init__(self, sequence_length, d_model=512):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=sequence_length, embedding_dim=d_model)

    def forward(self, x):
        x = self.embedding(x)  # 임베딩 레이어에 입력 데이터를 넣어 임베딩된 텐서를 얻습니다.
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, d_model=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(sequence_length, d_model)

    def forward(self)
        self.encoding.requires_grad = False
        pos = torch.arange(0, sequence_length)
        pos = pos.float().unsqueeze(dim=1)
        twopos = torch.arange(0, d_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (twopos / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (twopos / d_model)))
        return self.encoding

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, sequence_length, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.keys = nn.Linear(sequence_length, sequence_length)
        self.queries = nn.Linear(sequence_length, sequence_length)
        self.values = nn.Linear(sequence_length, sequence_length)
        self.layernorm = nn.LayerNorm(sequence_length)
        self.attention_drop = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(sequence_length, sequence_length)
        

    def forward(self, q, kv=None, mask=None): # y: key/value of encoder, mask: tensor
        residual = q # x: b, sequence_length, 512
        q = self.layernorm(q)
        # split keys, queries and values in num_heads
        queries = self.queries(q)
        queries = queries.view(queries.shape[0], self.num_heads, queries.shape[1], -1) # b, sequence_length, 512 -> b, 8(head), sequence_length, 64
        if kv is None: # self attention
            keys = self.keys(q)
            values = self.values(q)
        else: # encoder-decoder attention
            keys = self.keys(kv)
            values = self.values(kv)
        keys = keys.view(keys.shape[0], self.num_heads, keys.shape[1], -1) # b, sequence_length, 512 -> b, 8(head), sequence_length, 64
        values = values.view(values.shape[0], self.num_heads, values.shape[1], -1) # b, sequence_length, 512 -> b, 8(head), sequence_length, 64
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # Q*K^T, energy: b, 8, sequence_length, sequence_length
        scaling = self.sequence_length ** (1/2)
        attention = torch.softmax(energy, dim=-1) / scaling
        if mask is not None: # masked self attention
            attention = attention.masked_fill_(mask == 0, float('-inf'))
        attention = self.attention_drop(attention)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', attention, values) # att*V, out: b, 8, sequence_length, 64
        out = out.view(out.shape[0], out.shape[2], out.shape[1]*out.shape[3]) # b, 8, sequence_length, 64 -> b, sequence_length, 512
        out = self.projection(out)
        out = out + residual
        return out

class FeedForwardNetwork(nn.Sequential):
    def __init__(self, sequence_length, expansion=8, dropout_rate=0.1): # basic model ffn dimension = 2048
        super(FeedForwardNetwork, self).__init__()
        self.layernorm = nn.LayerNorm(sequence_length)
        self.linear1 = nn.Linear(sequence_length, expansion * sequence_length)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(expansion * sequence_length, sequence_length)
    def forward(self, x):
        residual = x # x: b, sequence_length, 512
        x = self.layernorm(x)
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual
        return out

# Define the transformer
class Transformer(nn.Sequential):
    def __init__(self, sequence_length, vocab_size, d_model=512, dropout_rate=0.1, num_heads=8, expansion=8, depth=8):
        super(Transformer, self).__init__()
        self.ie = InputEmbedding(sequence_length=sequence_length, d_model=d_model)
        self.pe = PositionalEncoding(sequence_length=sequence_length, d_model=d_model)
        self.mha = MultiHeadAttention(sequence_length=sequence_length, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn = FeedForwardNetwork(sequence_length=sequence_length, expansion=expansion, dropout_rate=dropout_rate)
        self.layer_norm = nn.LayerNorm(sequence_length)
        self.linear = nn.Linear(d_model, vocab_size)
        self.sequence_length = sequence_length
        self.depth = depth
    
    def forward(self, x, y): # x: encoder input, y: decoder input
        x = self.ie(x)
        x = x + self.pe(x)
        y = self.ie(y)
        y = y + self.pe(y)
        for i in range(self.depth):
            x = self.mha(q=x)
            x = self.ffn(x)
        Mask = torch.tril(torch.ones(self.sequence_length, self.sequence_length))
        for i in range(self.depth):
            y = self.mha(q=y, Mask=Mask) # masked self-head attention
            y = self.mha(q=y, kv=x) # encoder-decoder attention
            y = self.ffn(y)
        # Linear layer for classification
        output = self.linear(y) # y: b, sequence_length, 512 -> b, sequence_length, vocab_size
        return output