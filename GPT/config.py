from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # 50,000 BPE + 256 + 1 
    d_embed: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    n_layer: float = 12
    bias: bool = True
    n_head: int = 12
    device: str = 'cpu'
    flash_attention: bool = True

gpt2Config = GPTConfig(
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
)

gpt2MediumConfig = GPTConfig(
    n_layer=24, 
    n_head=16, 
    d_embed=1024
)

gpt2LargeConfig = GPTConfig(
    n_layer=36, 
    n_head=20, 
    d_embed=1280
)

gpt2XlConfig = GPTConfig(
    n_layer=48, 
    n_head=25, 
    d_embed=1600
)