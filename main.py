from transformer import Embedder, FeedForwardNetwork, Attention, TransformerEncoder, TransformerDecoder
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

zer = torch.full(
    size=(3, 3),
    fill_value=float('inf')
)
tri = torch.triu(zer, diagonal=1).unsqueeze(0).expand(3, -1, -1)

enc = TransformerEncoder(
    d_model=4,
    num_head=2, 
    d_ff=6,
    device=dev
)

dec = TransformerDecoder(
    d_model=4,
    num_head=2, 
    d_ff=6,
    device=dev
)

r = torch.rand(
    size=(3, 3, 4)
)

def create_causal_mask(d_model: int, seq_len: int, batch_size: int):
    zer = torch.full(
        size=(seq_len, d_model),
        fill_value=float('-inf')
    )
    tri = torch.triu(zer, diagonal=1).unsqueeze(0).unsqueeze(1)
    return tri

print(dec(a_emb).to(dev))