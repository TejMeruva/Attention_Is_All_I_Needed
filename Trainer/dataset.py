import torch
import tiktoken
import torch

class TextFile:
    def __init__(self, 
                 file_path: str,
                 batch_size: int, 
                 token_count: int,
                 device: str = 'cpu',
                 max_tokens = None
                 ):
        self.B = batch_size
        self.T = token_count

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
        self.current_pos = 0

    def next_batch(self):
        if (self.current_pos + self.B * self.T > len(self.tokens)): self.current_pos = 0
        batch_tokens =  self.tokens[self.current_pos: self.current_pos + self.B * self.T + 1]
        if len(batch_tokens) % self.B != 0: self.current_pos = 0
        x = batch_tokens[:-1].view(self.B, -1)
        y = batch_tokens[1:].view(self.B, -1) 

        if self.current_pos + (self.B * self.T) < self.tokens.size(0):
            self.current_pos += (self.B * self.T)
        else:
            self.current_pos = 0
        return x, y
            
