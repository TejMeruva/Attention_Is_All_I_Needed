from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import math
from GPT.config import *
from transformers import GPT2LMHeadModel
from tqdm import tqdm

# learnt from AK's video that taking a top-down approach is smarted here.
# so i will go from GPT -> Attn, Mlp

# I am using the same scheme as the original GPT-2 on Hugging Face. 
# this means same variable names as well.

class GPT2Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.c_attn = nn.Linear(
            in_features=config.d_embed, 
            out_features=config.d_embed * 3,
            bias = config.bias
        )

        self.c_proj = nn.Linear(
            in_features=config.d_embed, 
            out_features=config.d_embed,
            bias = config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    @staticmethod
    def causal_mask(*size):
        op = torch.full(
            size=(size[1], size[2]),
            fill_value=float('-inf')
        )

        op = torch.triu(op, diagonal=1)
        return op.unsqueeze(0).expand(size[0], -1, -1)

    def forward(self, x: torch.Tensor):
        b, l, d = x.size()

        # splitting into q, k, v
        q, k, v = self.c_attn(x.float()).split(self.config.d_embed, 2)

        # splitting into heads
        q = q.view(b, l, self.config.n_head, d // self.config.n_head).transpose(1, 2)
        k = k.view(b, l, self.config.n_head, d // self.config.n_head).transpose(1, 2)
        v = v.view(b, l, self.config.n_head, d // self.config.n_head).transpose(1, 2)

        att = (q @ k.transpose(-1, -2)) * (1/math.sqrt(k.size(-1)))
        # print(att.size())
        # causal_mask = self.causal_mask(a, b, c)
        # print(att.shape)
        # att += causal_mask
        att = att.masked_fill(self.bias[:,:,:l,:l] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        att = self.attn_dropout(att)
        att = att @ v

        att = att.transpose(1, 2).contiguous().view(b, l, d)
        att = self.resid_dropout(self.c_proj(att))
        return att

class GPT2MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=config.d_embed,
            out_features= config.d_embed * 4,
            bias=config.bias
        )
        self.act = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=config.d_embed * 4,
            out_features= config.d_embed,
            bias=config.bias
        )
        self.dropout = nn.Dropout(p = config.dropout)


    def forward(self, x: torch.Tensor):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_embed)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_embed)
        self.mlp = GPT2MLP(config)       

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

class GPT2(nn.Module):
    def __init__(self, config = GPTConfig()): #default config for gpt2 mini (124M)
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_embed
        ),

        # unlike the original transformer, even the positional embedding is learnt.
        wpe = nn.Embedding(
            num_embeddings=config.block_size, 
            embedding_dim=config.d_embed
        ),

        drop = nn.Dropout(p = config.dropout),

        h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(normalized_shape=config.d_embed)
        ))
        self.lm_head = nn.Linear(in_features=config.d_embed, out_features=config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        dev = idx.device
        self.to(dev)
        assert t <= self.config.vocab_size

        te = self.transformer.wte(idx)

        pos = torch.arange(0, t, device=dev, dtype=torch.long)
        pe = self.transformer.wpe(pos)

        x = self.transformer.drop(pe + te)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x[:, [-1], :])
        return x
    
    @classmethod
    def from_pretrained(self, modelName: str):
        model = GPT2()
        match modelName:
            case 'gpt2':
                model = GPT2(gpt2Config)
            case 'gpt2-medium':
                model = GPT2(gpt2Medium)
            case 'gpt2-large':
                model = GPT2(gpt2Large)
            case 'gpt2-xl':
                model = GPT2(gpt2Xl)
            case _:
                raise Exception(f'requested pre-trained model {modelName} not available.')
        
        stateDict = model.state_dict()
        modelHf = GPT2LMHeadModel.from_pretrained(modelName)
        stateDictHf = modelHf.state_dict()

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for key in stateDictHf.keys():
            if True in [key.endswith(x) for x in transposed]:
                with torch.no_grad():
                    assert stateDict[key].shape == stateDictHf[key].t().shape, f'shape of {key} : {stateDict[key].shape} is not equal to the required shape {stateDictHf[key].t().shape}'
                    stateDict[key].copy_(stateDictHf[key].t())
            else:
                assert stateDict[key].shape == stateDictHf[key].shape, f'shape of {key} : {stateDict[key].shape} is not equal to the required shape {stateDictHf[key].shape}'
                with torch.no_grad():
                    stateDict[key].copy_(stateDictHf[key])

        return model

    def param_count(self):
        s = 0
        for param in self.parameters():
            s += param.numel()
        s -= self.transformer.wpe.weight.numel()
        return s
    
    @torch.no_grad()
    def generate(self, idx: list, max_tokens: int = 30, k: int = 50, progress_bar = False):
        wrapper = lambda x: x
        if progress_bar:
            wrapper = tqdm
        for _ in wrapper(range(max_tokens)):
            logits = self(idx.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze()
            # print(probs)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next])

        return idx.tolist()
            
        
    # def generate(self, inpTokens: list, max_tokens):
    #     inpTokens = torch.tensor(inpTokens, dtype=torch.long, device=self.config.device)
    #     op = self.to(self.config.device)(inpTokens)
    #     op = F.softmax(op, dim=-1)
    #     return op.squeeze().argsort()[:50]
        