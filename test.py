from GPT.model import GPT2Attention, GPTConfig, GPT2
import torch
from AKGPT import GPT
from AKGPT import GPTConfig as gc

custConfig = GPTConfig(n_head=2, d_embed=4)
defConfig = GPTConfig(dropout=0)

att = GPT2Attention(custConfig)

a = torch.vstack([torch.tensor([1, 2, 3, 4]) for _ in range(3)]).unsqueeze(0).expand(3, -1, -1)
# print(a)
# print(att(a))

def causal_mask(d_embed: int, seq_len: int, batch_size: int):
    op = torch.full(
        size=(seq_len, d_embed),
        fill_value=float('-inf')
    )

    op = torch.triu(op, diagonal=1)
    return op.unsqueeze(0).expand(batch_size, -1, -1)
# print(att())

model = GPT2(defConfig).to('mps')
akModel = GPT(gc(n_head=2, n_embd=4)).to('mps')

s1 = 0
for param in model.parameters():
    s1 += param.numel()

# s2 = 0
# for param in akModel.parameters():
#     s2 += param.numel()

a = torch.tensor([1, 2, 3], dtype=torch.long, device='mps').unsqueeze(0).expand(3, -1)
# print(model(a))
# print(akModel(a))