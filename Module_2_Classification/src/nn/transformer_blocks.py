import math
import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Normalize layer
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    FeedForwardBlock
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """
    Transform input text to embeddings.
    In the embedding layers, multiply those weights by sqrt(dmodel)[Paper 3.4]
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement Positional Encoding - inject some information about the relative or absolute position of the
    tokens in the sequence [Paper 3.5]
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    We employ a residual connection around each of
    the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is
    LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
    itself. [Paper 3.1]
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    Implement MultiHeadAttentionBlock. [Paper 3.2]
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Just apply the formula from the paper -- softmax(query * key / sqrt(d_k))  * value
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        result = torch.matmul(attention_scores, value)
        return result, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """
    EncoderBlock [Paper 3.1]
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Encoder
    Consists of N layers - N encoder blocks
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class ClassificationHead(nn.Module):
    """
    Classification Head.
    Output layer dimension is (d_model, 1) for binary classification task
    """

    def __init__(self, d_model) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Fully Connected Layer
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        output = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        output = self.fc(output)
        return output


class Transformer(nn.Module):
    """
    Classification model itself.
    Combines Encoder and Classification head
    """

    def __init__(
        self,
        encoder: Encoder,
        src_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        classification_head: ClassificationHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.classification_head = classification_head

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def classification(self, x):
        return self.classification_head(x)

    def forward(self, src, src_mask):
        encoded = self.encode(src, src_mask)
        return self.classification(encoded)
