from Trainer.dataset import TextFile
from tqdm import tqdm
from GPT.model import GPT2
import torch
from GPT.config import gpt2Config
import matplotlib.pyplot as plt
import joblib

data = TextFile('Trainer/data.txt', 4, 100, device='mps', max_tokens=2000)

# model = joblib.load('GPT2_124M_01.pkl')
model = GPT2(gpt2Config)
print(f'using device {model.config.device}')
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=3e-4
)

# print(data.tokens.size(0) // (data.B * data.T))
# import sys; sys.exit(0)

epochs = 200
for ind in tqdm(range(epochs)):
    model.train()
    x, y = data.next_batch()
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (ind + 1) % 100 == 0:
        print(f'epoch {ind + 1}: loss is {loss}')

joblib.dump(model, 'GPT2_124M_overfit.pkl')

    

