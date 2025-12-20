from transformer import Embedder
import torch 

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

embedder = Embedder(
    vocab_size=100, 
    d_embed=8,
    d_model=4,
    device=dev
)

a = torch.tensor([1., 2., 3., 5.], device=dev, dtype=torch.long).unsqueeze(0)
print(embedder(a))