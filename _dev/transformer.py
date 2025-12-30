import torch 
from torch import nn
from math import log, sqrt
import torch.nn.functional as F

class Embedder(nn.Module):
    '''
    Converts the tokenized vector to Embeddings and Encodes Position.
    '''
    def __init__(self,
                 vocab_size: int, 
                 d_embed: int,
                 d_model:int,
                 device = 'cpu',
                 dropout = 0.1
                 ):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.device = device
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_embed,
            device=self.device
        )

        self.scaling_factor = sqrt(self.d_model)

        self.proj = nn.Linear(
            in_features=d_embed,
            out_features=d_model,
            device=self.device
        )

        self.norm = nn.LayerNorm(normalized_shape=self.d_model, device=self.device)
        self.dropout = nn.Dropout(p=self.dropout)

    @staticmethod
    def get_pos_encoding(inp: torch.Tensor, d_model: int, device: str) -> torch.Tensor:
        # assert inp.dtype == torch.long, f'Input Tensor must of dtype torch.long, got {inp.dtype}'
        batch_size, seq_length, d_embed = inp.size()

        dim = torch.arange(0, d_model, 2, device=device).float() # 2i's 
        div_fac = torch.exp((-dim)/d_model * 4 * log(10))

        pos = torch.arange(0, seq_length, device=device).float() \
        .unsqueeze(1) 

        pe = torch.zeros(
            size=(seq_length, d_model),
            device=device
        )

        pe[:, 0::2] = torch.sin(pos * div_fac)
        pe[:, 1::2] = torch.cos(pos * div_fac)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        x_proj = self.proj(x_emb) * self.scaling_factor
        enc = self.get_pos_encoding(x_proj, self.d_model, device=self.device)
        x = x + enc
        x = self.dropout(x)
        x = self.norm(x)
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(
                self,
                d_model: int,
                d_ff: int,
                device = 'cpu'
                ):
    
        super().__init__()
    
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device 
        
        self.layer_stack = nn.Sequential(
            nn.Linear(
                in_features=self.d_model,
                out_features=self.d_ff,
                device=self.device
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=self.d_ff,
                out_features=self.d_model,
                device=self.device
            )
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
    
class Attention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_head: int,
            device= 'cpu'
            ):

        super().__init__()

        self.d_model = d_model
        self.device = device
        self.num_head = num_head
        self.d_head = d_model // num_head

        self.q_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model, 
            device= self.device
        )

        self.k_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model, 
            device= self.device
        )

        self.v_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model, 
            device= self.device
        )

        self.scaling_factor = float(1.0 / sqrt(self.d_head))

        self.output_proj = nn.Linear(
            in_features=self.d_model, 
            out_features=self.d_model,
            device=self.device
        )
    
    def forward(self, 
                x: torch.Tensor,
                key_value_states: torch.Tensor = None,
                att_mask: torch.Tensor = None
                ) -> torch.Tensor:
        
        batch_size, seq_length, d_model = x.size()

        Q_state = self.q_proj(x)
        
        
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            kv_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_len = x.size(1)
            K_state = self.k_proj(x)
            V_state = self.v_proj(x)

        Q_state = Q_state.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2) #divides the tensor for each head
        K_state = K_state.view(batch_size, kv_len, self.num_head, self.d_head).transpose(1, 2)
        V_state = V_state.view(batch_size, kv_len, self.num_head, self.d_head).transpose(1, 2)
        
        op = torch.matmul(Q_state, K_state.transpose(-1, -2)) * self.scaling_factor
        
        if att_mask is not None:
            op = op + att_mask

        op = F.softmax(op, dim=-1)
        op = torch.matmul(op, V_state)
        # print(op)

        #concat
        op = op.view(batch_size, d_model, seq_length).transpose(-1, -2)

        #linear 
        op = self.output_proj(op)

        return op
    
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_head: int,
            d_ff: int, 
            device = 'cpu',
            dropout = 0.1
            ):
        
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.device = device
        self.dropout = dropout

        self.att = Attention(
            d_model=self.d_model, 
            num_head=self.num_head, 
            device=self.device
        )

        self.dropout = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(
            normalized_shape=self.d_model, 
            device=self.device
        )

        self.ffn = FeedForwardNetwork(
            d_model=self.d_model, 
            d_ff=self.d_ff, 
            device=self.device
        )

        self.norm2 = nn.LayerNorm(
            normalized_shape=self.d_model, 
            device=self.device
        )

    def forward(
            self,
            x:torch.Tensor,
            padding_mask: torch.Tensor = None
            ):
        
        x_att = self.att.forward(x, att_mask=padding_mask)

        x_att = self.dropout(x_att)
        x_norm1 = self.norm1(x_att + x)

        x_ffn = self.ffn(x_norm1)

        x_ffn = self.dropout(x_ffn)
        x_norm2 = self.norm2(x_ffn + x_norm1)

        return(x_norm2)
    
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_head: int,
            d_ff: int,
            device = 'cpu',
            dropout = 0.1
            ):

        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.device = device
        self.dropout = dropout

        self.att1 = Attention(
            d_model = self.d_model, 
            num_head= self.num_head, 
            device= self.device
        )

        self.dropout = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(
            normalized_shape=self.d_model, 
            device=self.device
        )

        self.att2 = Attention(
            d_model = self.d_model, 
            num_head= self.num_head, 
            device= self.device
        )

        self.norm2 = nn.LayerNorm(
            normalized_shape=self.d_model, 
            device=self.device
        )

        self.ffn = FeedForwardNetwork(
            d_model=self.d_model, 
            d_ff=self.d_ff, 
            device=self.device
        )

        self.norm3 = nn.LayerNorm(
            normalized_shape=self.d_model, 
            device=self.device
        )

    @staticmethod
    def create_causal_mask(d_model: int, seq_len: int, batch_size: int, device='cpu'):
        zer = torch.full(
            size=(seq_len, d_model),
            fill_value=float('-inf')
        )
        tri = torch.triu(zer, diagonal=1).unsqueeze(0).expand(batch_size, -1, -1)
        return tri.to(device)

    def forward(
            self, 
            x: torch.Tensor,
            padding_mask:torch.Tensor = None,
            cross_input: torch.Tensor = None
            ):
        
        batch_size, seq_len, d_model = x.size()
        
        causal_mask = self.create_causal_mask(
            d_model=self.d_model, 
            seq_len=seq_len, 
            batch_size=batch_size,
            device=self.device
        )

        x_att1 = self.att1(
            x=x, 
            att_mask=causal_mask
        )

        x_att1 = self.dropout(x_att1)
        x_norm1 = self.norm1(x_att1 + x)

        if cross_input is not None:
            x_att2 = self.att2(
                x=x_norm1, 
                key_value_states=cross_input, 
                att_mask=padding_mask
            )
        else: 
            x_att2 = self.att2(
                x=x_norm1,
                att_mask=padding_mask
            )

        x_att2 = self.dropout(x_att2)
        x_norm2 = self.norm2(x_att2 +  x_norm1)

        x_ffn = self.ffn(x_norm2)

        x_ffn = self.dropout(x_ffn)
        x_norm3 = self.norm3(x_ffn + x_norm2)

        return x_norm3

class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_head: int,
            d_ff: int,
            device = 'cpu',
            dropout = 0.1
            ):

        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.device = device
        self.dropout = dropout
    


