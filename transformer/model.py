import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) #bias is a learnable parameter

    def forward(self, x):
        #x: (batch_size, seq_len, hidden_size)
        #keepdim=True is used to keep the dimension of the tensor for broadcasting
        mean = x.mean(dim=-1, keepdim=True) #mean: (batch_size, seq_len, 1)
        #keepdim=True is used to keep the dimension of the tensor for broadcasting
        std = x.std(dim=-1, keepdim=True) #std: (batch_size, seq_len, 1)
        #eps is used to prevent division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)# w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)# w2 and b2

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #older way to do this
        # for pos in range(seq_len):
        #     for i in range(d_model):
        #         if i % 2 == 0:
        #             pe[pos][i] = math.sin(pos / (10000 ** (i/d_model)))
        #         else:
        #             pe[pos][i] = math.cos(pos / (10000 ** ((i-1)/d_model)))
        #new way to do this
        #create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  #sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        #create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  #cos(position * (10000 ** (2i / d_model)))
        #create a matrix of shape (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)
        #register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch_size, seq_len, d_model) + (1, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.d_model = features
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # (batch_size, seq_len, d_model) + (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return x + self.dropout(sublayer(self.norm(x)))
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model #Embedding vector size
        self.h = h #Number of heads
        #make sure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.head_dim = d_model // h #Dimension of each head
        self.w_q = nn.Linear(d_model, d_model) #Query matrix Wq
        self.w_k = nn.Linear(d_model, d_model) #Key matrix Wk
        self.w_v = nn.Linear(d_model, d_model) #Value matrix Wv
        self.w_o = nn.Linear(d_model, d_model) #Output matrix Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout = None):
        #query: (batch_size, h, seq_len, d_k)
        #key: (batch_size, h, seq_len, d_k)
        #value: (batch_size, h, seq_len, d_k)
        #mask: (batch_size, seq_len, seq_len)
        #key.transpose(-2, -1) 
        d_k = query.size(-1)
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)# (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        output = attention_scores @ value
        return output, attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) #(batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) #(batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) #(batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) 
        
        batch_size, seq_len, d_model = query.size()
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        #calculate the attention weights
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
       
        #multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # self.residual_connection_1 = ResidualConnection(features, dropout)
        # self.residual_connection_2 = ResidualConnection(features, dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask=None):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
        
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)  
        src = self.src_embed(src)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        src = self.src_pos(src)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        tgt = self.tgt_pos(tgt)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int=6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters with Xavier uniform initialization
    # Xavier initialization helps maintain the variance of activations throughout the network,
    # which prevents the gradients from vanishing or exploding during training.
    # This is particularly important for deep networks like transformers.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
