import torch
import torch.nn as nn
from torchsummary import summary
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
authors came up with Positional Encoding, wich is summed to 
the original embedding vectors.
"""
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        # Create a tensor of zeroes to fill in the following steps
        pe = torch.zeros(seq_len, d_model)
        # Create a positions vector of shape seq_len. We use .arange(), 
        # begining at 0 and finising in the max of the seq_len
        position = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)
        # We create the division term of the formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        # Apply sine and cosine. The sine is applied to even numbers; cosine to odd ones
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Now we need to make it applicable to batches. To do so, we need 
        # to add an extra dimension in the first position. We do so by using
        # unsqueeze at position 0
        pe = pe.unsqueeze(0)

        # We need to register the tensor in the buffer of the module (kind of remember it).
        self.register_buffer('pe', pe)

    def forward(self, x):
        # We add the positional encoding to the input tensor. We slice it to match
        # the dimensions of the word embedding. Remember the dimensions are: 
        # 0: Batch
        # 1: Embeddings
        # 2: Dimension
        # We take dim 1 (Embeddings) and align them with the shape dim 1 (length) of x (the 
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


