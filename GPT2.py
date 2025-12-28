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
    bias: bool = True

class GPT2Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

class GPT2MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=config.d_embed,
            out_features= config.d_embed * 4,
            bias=config.bias
        )
        self.act = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=config.d_embed * 4,
            out_features= config.d_embed,
            bias=config.bias
        )
        self.dropout = nn.Dropout(p = config.dropout)


    def forward(self, x: torch.Tensor):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_embed)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_embed)
        self.mlp = GPT2MLP(config)        

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

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

        h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
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
            x = block(x)

        x = self.transformer.ln_f(x)
        x = self.transformerlm_head(x)
        return x