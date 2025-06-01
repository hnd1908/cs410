import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.weight * normalized + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        assert d_model % head == 0, "d_model must be divisible by head"
        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout=nn.Dropout):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / d_k**0.5

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        return (scores @ value), scores

    def forward(self, q, k , v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.size(0), query.size(1), self.head, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.head, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.head, self.d_k).transpose(1, 2)

        x, scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        self.attention_scores = scores
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head * self.d_k)

        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention = self_attention_block
        self.ff = feed_forward_block
        self.residual1 = ResidualConnection(features, dropout)
        self.residual2 = ResidualConnection(features, dropout)

    def forward(self, x, src_mask):
        x = self.residual1(x, lambda x: self.attention(x, x, x, src_mask))
        x = self.residual2(x, self.ff)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.ff = feed_forward_block
        self.residual1 = ResidualConnection(features, dropout)
        self.residual2 = ResidualConnection(features, dropout)
        self.residual3 = ResidualConnection(features, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual3(x, self.ff)
        return x        
    
class Decoder(nn.Module):
        
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.linear(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = src.long()  # Ensure indices are integers
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = tgt.long()  # Ensure indices are integers
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    layers = nn.ModuleList([
        EncoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])
    
    encoder = Encoder(d_model, layers)
    
    layers = nn.ModuleList([
        DecoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(N)
    ])
    
    decoder = Decoder(d_model, layers)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer