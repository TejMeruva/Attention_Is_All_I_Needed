# importing libs/modules
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import seaborn as sns
import os
import math
import time
import joblib
from Trainer.dataset import NumPyFolder
from tqdm import tqdm
from GPT.model import GPT2
from GPT.config import GPTConfig
import torch
from GPT.config import gpt2Config
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# graph formatting
plt.rcParams['font.family'] = 'SF Mono'
plt.rcParams['font.size'] = 10
plt.figure(figsize=(40, 16))   # options: 'serif', 'sans-serif', 'monospace'
colors = ["#1D2E50", '#334879', '#8EB4E3', '#D7E0F4']
colorsSorted = ["#1D2E50", '#334879', '#8EB4E3', '#D7E0F4'][::-1]
sns.set_palette(colors)

# distributed data parallel
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), f'cuda not available'
    init_process_group('nccl')  # NVIDIA Collective Communications Library
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    is_master_process = (ddp_local_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    is_master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    print(f'using device: {device}')


# getting the training data

data = NumPyFolder(
    file_path='fineweb_data',
    batch_size=4,
    token_count=1024,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
    device=device
)

# manual seed
torch.manual_seed(69)
if torch.cuda.is_available():
    torch.cuda.manual_seed(69)

# using TF32
torch.set_float32_matmul_precision('medium')  # using TF32 instead of FP32.

# create model
model = GPT2(GPTConfig(
    device=device,
    vocab_size=50304,
    flash_attention=True
)).to(device)
if ddp:
    model = DDP(
        module=model,
        device_ids=[ddp_local_rank]
    )
raw_model = model.module if ddp else model

print(f'model on device: {model.config.device}')

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

    decay = [p for n, p in param_dict.items() if p.dim() >= 2]
    non_decay = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': non_decay, 'weight_decay': 0.0}
    ]

    using_fused = model.config.device == 'mps'

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=using_fused
    )
    num_decay = sum([p.numel() for p in decay])
    num_non_decay = sum([p.numel() for p in non_decay])
    print(f'{num_decay} parameters using weight decay')
    print(f'{num_non_decay} parameters NOT using weight decay')
    print(f'AdamW fused is being used: {using_fused}')
    return optimizer


optimizer = configure_optimizer(raw_model, 0.1)

losses = []

batch_tokens = 2**19
micro_batch_tokens = (data.B * data.T)
gradient_accumulation_steps = batch_tokens // (
    micro_batch_tokens * ddp_world_size)
steps = 1

if is_master_process:
    print(f'total tokens in data: {len(data.tokens)}')
    print(f'batches needed: {len(data.tokens) / batch_tokens}')
    print(f'tokens per batch: {batch_tokens}')
    print(f'tokens per micro-batch: {micro_batch_tokens}')
    print(f'ddp world size: {ddp_world_size}')
    print(f'gradient acculumation steps: {gradient_accumulation_steps}')
    print(f'total batches: {steps}')

for step in range(steps):
    model.train()
    t0 = time.time()
    loss_accum = 0
    optimizer.zero_grad()
    print(f'batch {step+1}')
    for sub_step in tqdm(range(gradient_accumulation_steps)):
        x, y = data.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss/gradient_accumulation_steps
        loss_accum += loss.detach().item()
        if ddp:
            model.require_backward_grad_sync = (
                sub_step == (gradient_accumulation_steps-1))
            # averages out the gradients from all steps
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    optimizer.step()
    losses.append(loss_accum)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    torch.mps.synchronize()
    t1 = time.time()
    tok_rate = (data.B * data.T * gradient_accumulation_steps *
                ddp_world_size) / (t1 - t0)
    if is_master_process:
        print(f'epoch ({step}/{steps})\ttime for epoch: {t1 - t0}\ttok/sec: {tok_rate}\tlr: {lr}\tloss: {loss_accum.item()}\tnorm: {norm}')

if ddp:
    destroy_process_group()

# saving the loss v/s epochs
fig, ax = plt.subplots(1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('CE loss')
ax.set_title('Loss v/s batch')
fig.savefig('loss.png', dpi=300)

joblib.dump(model, 'GPT2_124M_02.pkl')
