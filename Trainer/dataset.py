import torch
import tiktoken
import torch
from datasets import load_dataset
import os
import numpy as np


class NumPyFolder:
    def __init__(self,
                 file_path: str,
                 batch_size: int,
                 token_count: int,
                 ddp_rank: int,
                 ddp_world_size: int,
                 device: str = 'cpu'
                 ):
        self.device = device
        self.file_path = file_path
        self.B = batch_size
        self.T = token_count
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        self.shards = sorted(os.listdir(file_path))
        self.current_shard = -1
        self.current_pos = self.B * self.T * self.ddp_rank
        self.load_next_shard()

    def load_next_shard(self):
        if self.current_shard + 1 >= len(self.shards):
            self.current_shard = -1
        arr = np.load(os.path.join(
            self.file_path,
            self.shards[self.current_shard + 1]
        ),
            mmap_mode='r')
        self.current_shard += 1
        self.tokens = torch.from_numpy(arr).type(torch.int32)

    def next_batch(self):
        batch_tokens = self.tokens[self.current_pos:
                                   self.current_pos + self.B * self.T + 1]
        assert len(self.tokens) % (
            self.B * self.T) == 0, f'tokens in shard {len(self.tokens)}, not divisible by micro batch size {self.B * self.T}'
        x = batch_tokens[:-1].view(self.B, (len(batch_tokens) - 1) // self.B)
        y = batch_tokens[1:].view(self.B, (len(batch_tokens) - 1) // self.B)

        step = self.B * self.T * self.ddp_world_size
        self.current_pos += step

        if self.current_pos + step > self.tokens.size(0):
            self.load_next_shard()
            self.current_pos = self.B * self.T * self.ddp_rank
        return x.to(self.device), y.to(self.device)


class TextFile:
    def __init__(self,
                 file_path: str,
                 batch_size: int,
                 token_count: int,
                 ddp_rank: int,
                 ddp_world_size: int,
                 device: str = 'cpu',
                 max_tokens=None
                 ):
        self.B = batch_size
        self.T = token_count
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        self.tokenizer = tiktoken.get_encoding('gpt2')

        with open(file_path, 'r') as file:
            text = file.read()
            tokens = torch.tensor(
                self.tokenizer.encode(text),
                device=device,
                dtype=torch.long)

        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        self.tokens = tokens
        self.current_pos = self.ddp_rank * self.B * self.T

    def next_batch(self):
        batch_tokens = self.tokens[self.current_pos:
                                   self.current_pos + self.B * self.T + 1]
        x = batch_tokens[:-1].view(self.B, (len(batch_tokens) - 1) // self.B)
        y = batch_tokens[1:].view(self.B, (len(batch_tokens) - 1) // self.B)

        step = self.B * self.T * self.ddp_world_size
        self.current_pos += step

        if self.current_pos + step > self.tokens.size(0):
            self.current_pos = self.B * self.T * self.ddp_rank
        return x, y
