from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import math


# learnt from AK's video that taking a top-down approach is smarted here.
# so i will go from GPT -> Attn, Mlp

# I am using the same scheme as the original GPT-2 on Hugging Face. 
# this means same variable names as well.

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # 50,000 BPE + 256 + 1 
    d_embed: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    n_layer: float = 12
    bias: bool = True
    n_head: int = 12
    device: str = 'cpu'
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
    def __init__(self, config : GPTConfig):
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
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        # config = GPTConfig(**config_args)
        model = GPT2(GPTConfig())
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    @torch.no_grad()
    def generate(self, idx: list, max_tokens: int = 30, k: int = 50):
        for _ in range(max_tokens):
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
        