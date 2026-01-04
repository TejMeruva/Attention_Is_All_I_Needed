import torch
import tiktoken
import torch

class TextFile:
    def __init__(self, 
                 file_path: str,
                 batch_size: int, 
                 token_count: int,
                 ddp_rank: int, 
                 ddp_world_size: int,
                 device: str = 'cpu',
                 max_tokens = None
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
        batch_tokens =  self.tokens[self.current_pos: self.current_pos + self.B * self.T + 1]
        x = batch_tokens[:-1].view(self.B, (len(batch_tokens) - 1) // self.B)
        y = batch_tokens[1:].view(self.B, (len(batch_tokens) - 1) // self.B) 

        step = self.B * self.T * self.ddp_world_size
        self.current_pos += step

        if self.current_pos + step > self.tokens.size(0) :
            self.current_pos = self.B * self.T * self.ddp_rank
        return x, y
            
