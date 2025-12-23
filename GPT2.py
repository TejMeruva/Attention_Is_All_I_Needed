from dataclasses import dataclass
import torch
from torch import nn


# learnt from AK's video that taking a top-down approach is smarted here.
# so i will go from GPT -> Attn, Mlp

# I am using the same scheme as the original GPT-2 on Hugging Face. 
# this means same variable names as well.

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # 50,000 BPE + 256 + 1 
    d_embed: int = 768
    block_size: int = 1024
    dropout: float = 0.1
    n_layer: float = 12

class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

class GPT(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_embed
        ),

        # unlike the original transformer, even the positional embedding is learnt.
        wpe = nn.Embedding(
            num_embeddings=config.block_size, 
            embedding_dim=config.d_embed
        ),

        drop = nn.Dropout(p = config.dropout),

        h = nn.ModuleList([GPT2Block() for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(normalized_shape=config.d_embed)
        ))
        self.lm_head = nn.Linear(in_features=config.d_embed, out_features=config.vocab_size, bias=False)

    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        dev = idx.device
        assert t <= self.config.vocab_size

        te = self.transformer.wte(idx)

        pos = torch.arange(0, t, device=dev, dtype=torch.long)
        pe = self.transformer.wpe(pos)

        x = self.transformer.drop(pe + te)

        for block in self.transformer.h:
            x = block()

        x = self.transformer.ln_f(x)
        x = self.transformerlm_head(x)
        return x