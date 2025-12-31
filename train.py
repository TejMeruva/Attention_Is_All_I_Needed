from Trainer.dataset import TextFile
from tqdm import tqdm
from GPT.model import GPT2
from GPT.config import GPTConfig
import torch
from GPT.config import gpt2Config
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import joblib
import time
import math
import seaborn as sns

plt.rcParams['font.family'] = 'SF Mono'
plt.rcParams['font.size'] = 10
plt.figure(figsize=(40, 16))   # options: 'serif', 'sans-serif', 'monospace'
colors =[ "#1D2E50",'#334879','#8EB4E3','#D7E0F4']
colorsSorted = [ "#1D2E50",'#334879','#8EB4E3','#D7E0F4'][::-1]
sns.set_palette(colors)

torch.manual_seed(42)
device = 'mps'
data = TextFile('Trainer/data.txt', 4, 1024, device=device)

torch.set_float32_matmul_precision('medium') # using TF32 instead of FP32.

# model = joblib.load('GPT2_124M_01.pkl')
model = GPT2(GPTConfig(device=device, vocab_size=50304, flash_attention=True))
# model = torch.compile(model)
print(f'using device {model.config.device}')

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it <= warmup_steps:
        return (it+1)/(warmup_steps) * max_lr
    if (it <= max_steps):
        decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8
)

# print(data.tokens.size(0) // (data.B * data.T))
# import sys; sys.exit(0)
losses = []
steps = len(data.tokens) // (data.B * data.T)
steps = 20
print(f'total num of epochs: {steps}')

prog_bar = False
wrapper = lambda x: x
if prog_bar: wrapper = tqdm
for step in wrapper(range(steps)):
    model.train()    
    t0 = time.time()
    x, y = data.next_batch()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    losses.append(loss.item())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    tok_rate = (data.B * data.T) / (t1 - t0)
    print(f'epoch ({step}/{steps})\ttime for epoch: {t1 - t0}\ttok/sec: {tok_rate}\tlr: {lr}\tloss: {loss.item()}\tnorm: {norm}')
    
    

#saving the loss v/s epochs
fig, ax = plt.subplots(1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('CE loss')
ax.set_title('Loss v/s batch')
fig.savefig('loss.png', dpi=300)

joblib.dump(model, 'GPT2_124M_02.pkl')

    

