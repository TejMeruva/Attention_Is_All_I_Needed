import torch 
from torch import nn
from math import log, sqrt

class Embedder(nn.Module):
    '''
    Converts the tokenized vector to Embeddings and Encodes Position.
    '''
    def __init__(self,
                 vocab_size: int, 
                 d_embed: int,
                 d_model:int,
                 device = 'cpu',
                 dropout = 0.1
                 ):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.device = device
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_embed,
            device=self.device
        )

        self.scaling_factor = sqrt(self.d_model)

        self.proj = nn.Linear(
            in_features=d_embed,
            out_features=d_model,
            device=self.device
        )

        self.norm = nn.LayerNorm(normalized_shape=self.d_model, device=self.device)
        self.dropout = nn.Dropout(p=self.dropout)

    @staticmethod
    def get_pos_encoding(inp: torch.Tensor, d_model: int, device: str) -> torch.Tensor:
        # assert inp.dtype == torch.long, f'Input Tensor must of dtype torch.long, got {inp.dtype}'
        batch_size, seq_length, d_embed = inp.size()

        dim = torch.arange(0, d_model, 2, device=device).float() # 2i's 
        div_fac = torch.exp((-dim)/d_model * 4 * log(10))

        pos = torch.arange(0, seq_length, device=device).float() \
        .unsqueeze(1) 
        
        pe = torch.zeros(
            size=(seq_length, d_model),
            device=device
        )

        pe[:, 0::2] = torch.sin(pos * div_fac)
        pe[:, 1::2] = torch.cos(pos * div_fac)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        x_proj = self.proj(x_emb) * self.scaling_factor
        enc = self.get_pos_encoding(x_proj, self.d_model, device=self.device)
        x = self.norm(x+enc)
        x = self.dropout(x)
        return x
    
