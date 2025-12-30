from Trainer.dataset import TextFile
from tqdm import tqdm
from GPT.model import GPT2
from GPT.config import GPTConfig
import torch
from GPT.config import gpt2Config
import matplotlib.pyplot as plt
import joblib
import time

torch.manual_seed(69)

device = 'mps'

data = TextFile('Trainer/data.txt', 8, 512, device=device)

torch.set_float32_matmul_precision('medium') # using TF32 instead of FP32.

# model = joblib.load('GPT2_124M_01.pkl')
model = GPT2(GPTConfig(device=device))
modle = torch.compile(model)
print(f'using device {model.config.device}')
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=3e-4
)

# print(data.tokens.size(0) // (data.B * data.T))
# import sys; sys.exit(0)
losses = []
epochs = len(data.tokens) // (data.B * data.T)
print(f'total num of epochs: {epochs}')

prog_bar = False
wrapper = lambda x: x
if prog_bar: wrapper = tqdm
for ind in wrapper(range(epochs)):
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    x, y = data.next_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    # losses.append(loss.item())
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    tok_rate = (data.B * data.T) / (t1 - t0)
    print(f'epoch ({ind}/{epochs})\ttime for epoch: {t1 - t0}\ttok/sec: {tok_rate}')
    
    if (ind + 1) % 100 == 0:
        print(f'epoch {ind + 1}: loss is {loss}')

#saving the loss v/s epochs
fig, ax = plt.subplots(1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('CE loss')
ax.set_title('Loss v/s batch')
fig.savefig('loss.png', dpi=300)

joblib.dump(model, 'GPT2_124M_02.pkl')

    

