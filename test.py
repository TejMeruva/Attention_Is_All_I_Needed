from Trainer.dataset import NumPyFolder
import numpy as np

data = NumPyFolder(
    file_path='fineweb_data',
    batch_size=16,
    token_count=1024,
    ddp_rank=0,
    ddp_world_size=1,
    device='mps'
)
for _ in range(1000):
    print(*map(lambda x: x.shape, data.next_batch()))
