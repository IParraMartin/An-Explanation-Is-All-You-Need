import torch
import torch.nn as nn
import math


"""
Input Embeddings are the first step to build a transformer.
We set up the dimension of the embeddings (d_model),
and the vocab size (vocan_size). Then we use an embedding
layer from torch.nn and set num_embeddings and embedding_dim
"""
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.voca_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
    
    def forward(self, x):
        # Multiply the embeddings by the squared root of the d_model (Vaswani et al. 2017).
        return self.embedding(x) * math.sqrt(self.d_model)


"""
Because the attention mechanisms are position invariant, we need
to encode the information of word order in some way. The 
authors came up with positional encoding, which is summed to 
the original embeddings.
"""
class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        # Create a tensor of zeros to fill in
        pe = torch.zeros(seq_len, d_model)
        # Create a positions vector of shape seq_len. We use .arange()
        position = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)
        # We create the division term of the formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        # Apply sine and cosine. The sine is applied to even numbers; cosine to odd numbers
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Now we need to make it applicable to batches. To do so, we need 
        # to add an extra dimension in the first position. We do so by using
        # .unsqueeze at position 0
        pe = pe.unsqueeze(0)

        # We need to register the tensor in the buffer of the module (kind of remember it).
        self.register_buffer('pe', pe)

    def forward(self, x):
        # We add the positional encoding to the input tensor. We slice it to match
        # the dimensions of the word embedding. Remember the dimensions are: [batch, pos_embedings, dimension]
        # We take dim 1 (pos_embeddings) and align them with the dim 1 of x (the 
        # actual word embeddings). We also make sure to make the positional
        # embeddings static (.requires_grad).
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

"""
We now design the Layer Normalization. We do this by passing an epsilon value (eps), 
which is added to avoid 0 division in the normalization operation.
"""
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        # We set alpha and bias as trainable params
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Get mean and std
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # Finally use the formula to normalize
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


"""
This is the Position Wise Feed-Forward Network. That's just a fancy name
for a simple neural network, with two layers and a Relu activation in between.
"""
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # This corresponds to W1 and B1 of Vaswani et al. (2017)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # This corresponds to W2 and B2

    def forward(self, x):
        # Pass the main function and return [d_model] dim
        return self.linear2(self.dropout(nn.ReLU(self.linear1(x))))


"""
This is probably the most important block: the famous Multi-Head Attention.
"""
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # We must be able to send equal dims to the different heads
        assert d_model % n_heads == 0, "Division between d_model and n_heads must be possible." 

        # d_k is the dim that each tensor will have to be parallelized in different heads
        self.d_k = d_model // n_heads

        # Set up the weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv

        # Set Wo, which is [(n_heads * self.d_v), d_model] = [d_model, d_model]
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        """
        COMMON QUESION: If d_k and d_v are the same dimensions,
        why do they have different names? d_v is the result of the last
        multiplication of the attention formula (which is by V; see 
        the original paper). However, in practice they are the same.
        """


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # We extact the d_k, which is the last dimension of Q, K, and V
        d_k = query.shape[-1]
        # We apply the self-attention scores formula. We transpose the last two dims to make the calculation possible
        # Transform: [Batch, n_heads, seq_len, d_k] -> [Batch, n_heads, seq_len, seq_len]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # This is for the masking
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # We apply softmax
        attention_scores = attention_scores.softmax(dim = -1) 
        # We add dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # We return a tupple with the attention and the self-attention for visualization
        return (attention_scores @ value), attention_scores
    

    def forward(self, q, k, v, mask):
        # Project the embeddings into the weight matrices [Batch, seq_len, d_model] -> [Batch, seq_len, d_model]
        query = self.w_q(q)
        key = self.w_k(k) 
        value = self.w_v(v) 

        """
        IMPORTANT:
        We need to divide Q, K and V to feed "pieces" to different heads (power of parallel processing!)
        Transform: [Batch, seq_len, d_model] -> [Batch, seq_len, n_heads, d_k] -> [Batch, n_heads, seq_len x d_k]

            - We don't want to split the batches: query.shape[0]
            - We don't want to split the sequence: query.shape[1]
            - We want to split the d_model (embeddings): self.n_heads, self.d_k
            - We want the transposition because we want each head to see the seq_len and d_k
        
        The transposition allows the model to process each head independently across the sequence length. Each head can 
        focus on different parts of the input sequence, enabling the model to capture various aspects of 
        the input data in parallel.
        """
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        
        # here we use the function we previously introduced as a static method
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Here we concatenate the information from the different heads. By multiplying n_heads and d_k, 
        # we effectively concatenate the d_k elements from each head for every sequence position into a single 
        # dimension, resulting in a tensor where the information from different heads is concatenated 
        # for each position in the sequence.
        # Transform: [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # Project x it in weights Wo and return it as [batch, seq_len, d_model]
        return self.w_o(x)
    

"""
Here we will build the residual connection component of the transformer. 
This will allow a better training and make some 'raw' input flow from layer to layer.
"""
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization()

    def forward(self, x, sublayer):
        # Normalize x, then pass it through a sublayer (any type), use the dropout term, and finally add x
        return x + self.dropout(sublayer(self.normalization(x)))


"""
Here's the encoder block that we will use to create the 
Encoder object (stacking of Encoder layers)
"""
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        # This is the multi-head attention
        self.self_attention_block = self_attention_block
        # This is the Point-wise feed forward network
        self.feed_forward_block = feed_forward_block
        # we pack two residual connections in a nn.ModuleList
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # In the first residual connection (idx: 0), we are using MultiHeadAttention, which takes Q, K, V and mask, and
        # the residual input. We add both and we pass that result to the second residual connection (idx: 1). This
        # makes the same operation but with the FeedForwardBlock.
        x = self.residual_connections[0](lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


"""
This is how we stack the Encoder block in several layers. This will be 
the main Encoder object. Because it has to be able to take several Encoder
blocks, we use nn.ModuleList as a parameter
"""
class Encoder(nn.Module):

    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        # We can create as many layers as we want
        self.n_layers = n_layers
        self.normalization = LayerNormalization()

    def forward(self, x, mask):
        # we iterate over n layers
        for layer in self.n_layers:
            x = layer(x, mask)
        # finally we normalize
        return self.normalization(x)
    

"""
This will be the Decoder block that will allow us to make several layers of it. 
We introduce cross attention, which is similar to multi-head attention but taken 
parameters from the encoder.
"""
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        # Initialize all the pieces
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Get three residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x = self.residual_connections[0](lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](lambda x: self.self_attention_block(x, encoder_out, encoder_out, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)


"""
This will be our main Decoder object that we will use to ensemble
all layers.
"""
class Decoder(nn.Module):

    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        # we set the n_layers parameter
        self.n_layers = n_layers
        self.normalization = LayerNormalization()

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # we iterate over n layers
        for layer in self.n_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        # finally we normalize
        return self.normalization(x)
        

"""
Final layer to convert logits to a probability distribution over all the vocabulary
of the target language (we are building the og transformer)
"""
class LastLinear(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.fc(x), dim=-1)
    

"""
This is the actual beast: the Transformer. We use all the blocks we have been 
carefully assembling to build the following class
"""
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_pos: PositionalEmbeddings, tgt_pos: PositionalEmbeddings, 
                 last_linear: LastLinear):
        super().__init__()
        # Set all the components we will need
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.last_linear = last_linear

    # We define an encode() function for the encoder.
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    # We define a decode() function for the decoder.
    def decode(self, enc_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)
    
    # We pass the outputs through the final linear layer
    def linear(self, x):
        return self.last_linear(x)


"""
We will finally create a function that allows us to ensemble everything
together and create the layers of encoders and decoders. This is what the
original paper calls Nx in the figure to the sides of the main figure.
"""
def build_transformer_model(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, 
                            tgt_seq_len: int, d_model: int= 512, n_layers: int = 6, 
                            n_heads: int = 8, dropout: float = 0.1, hidden_size: int = 2048):
    
    # Make the embedding layers for input and target
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Make the positional embeddings for input and target 
    # (in practice you can use the same for both src and tgt)
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, hidden_size, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward = FeedForwardBlock(d_model, hidden_size, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    """
    Now we pass those layers as the argumemtn to the main objects.
    Remember that our main classes for encoder and decoder take
    nn.ModuleList as arguments (aka, the layers stacked)
    """
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    last_layer = LastLinear(d_model, tgt_vocab_size)

    # Now, build our model using the transformer class we built
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        last_linear=last_layer
    )

    # Now we initialize the parameters with Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # FINALLY, return the transformer
    return transformer
