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

def configure_optimizer(model: torch.nn.Module, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    decay = [p for n, p in param_dict.items() if p.dim() >=2]
    non_decay = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params':decay, 'weight_decay':weight_decay},
        {'params':non_decay, 'weight_decay':0.0}
    ]

    using_fused = model.config.device == 'mps'

    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused= using_fused
        )
    num_decay = sum([p.numel() for p in decay])
    num_non_decay = sum([p.numel() for p in non_decay])
    print(f'{num_decay} parameters using weight decay')
    print(f'{num_non_decay} parameters NOT using weight decay')
    print(f'AdamW fused is being used: {using_fused}')
    return optimizer

optimizer = configure_optimizer(model, 0.1)

# print(data.tokens.size(0) // (data.B * data.T))
# import sys; sys.exit(0)
losses = []

# steps = 20


prog_bar = False
wrapper = lambda x: x

batch_tokens = 2**19 
micro_batch_tokens = (data.B * data.T)
gradient_accumulation_steps = batch_tokens //micro_batch_tokens
steps = 1

print(f'total tokens in data: {len(data.tokens)}')
print(f'batches needed: {len(data.tokens) / batch_tokens}')
print(f'tokens per batch: {batch_tokens}')
print(f'tokens per micro-batch: {micro_batch_tokens}')
print(f'gradient acculumation steps: {gradient_accumulation_steps}')
print(f'total batches: {steps}')

if prog_bar: wrapper = tqdm
for step in wrapper(range(steps)):
    model.train()    
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    print(f'batch {step+1}')
    for sub_step in tqdm(range(gradient_accumulation_steps)):
        x, y = data.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss/gradient_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    losses.append(loss_accum)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    torch.mps.synchronize()
    t1 = time.time()
    tok_rate = (data.B * data.T) / (t1 - t0)
    print(f'epoch ({step}/{steps})\ttime for epoch: {t1 - t0}\ttok/sec: {tok_rate}\tlr: {lr}\tloss: {loss_accum}\tnorm: {norm}')
    
    

#saving the loss v/s epochs
fig, ax = plt.subplots(1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('CE loss')
ax.set_title('Loss v/s batch')
fig.savefig('loss.png', dpi=300)

joblib.dump(model, 'GPT2_124M_02.pkl')

    

