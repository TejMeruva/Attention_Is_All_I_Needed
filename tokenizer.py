import tiktoken
from GPT.model import GPT2, GPTConfig
from AKGPT import GPT
from random import choice, seed
import torch

# torch.manual_seed(69)
# seed(69)

model = GPT2.from_pretrained('gpt2')


tokenizer = tiktoken.get_encoding('gpt2')
op = torch.tensor(tokenizer.encode('My name is'), device='mps')
# op = torch.tensor(tokenizer.encode('My name is'), device='mps').unsqueeze(0)

print(tokenizer.decode(model.generate(op, 30)))

# model = GPT.from_pretrained('gpt2').to('mps')
# print(tokenizer.decode(model.generate(op, max_new_tokens=30).squeeze().tolist()))