from transformer import Embedder, FeedForwardNetwork, Attention
import torch 

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

embedder = Embedder(
    vocab_size=100, 
    d_embed=8,
    d_model=4,
    device=dev
)

a = torch.tensor([1., 2., 3., 5.], device=dev, dtype=torch.long).unsqueeze(0)
a_emb = embedder(a)

ffn = FeedForwardNetwork(
    d_model=4, 
    d_ff= 6, 
    device=dev
)

att = Attention(
    d_model=4, 
    num_head=2, 
    device=dev
)

print(att.forward(
    x=a_emb
))