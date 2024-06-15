import torch
import torch.nn as nn
import math

from torchsummary import summary


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
        return self.embedding(x) * math.sqrt(self.d_model)

