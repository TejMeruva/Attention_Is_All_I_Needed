from torch import nn 
import torch.nn.functional as F
import torch
from math import log, sqrt

class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, 
                 d_embed: int, 
                 d_model: int,
                 dropout_p: float = 0.1,
                 dev = 'cpu'
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.dev = dev
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_embed,
            device=self.dev
        )
        self.projection = nn.Linear(
            in_features=d_embed,
            out_features=d_model,
            device=self.dev
        )
        self.scaling = float(sqrt(self.d_model))

        self.layerNorm = nn.LayerNorm(
            self.d_model,
            device=self.dev
        )
        
        self.dropout = nn.Dropout(p=dropout_p)

    @staticmethod # decorator that indicates that the following function doesn't operate on `self`
    def create_positional_encoding(seq_length:int, 
                                   d_model:int, 
                                   batch_size:int,
                                   dev = 'cpu'
                                   ):

        positions = torch.arange(seq_length, dtype=torch.long, device=dev)\
            .unsqueeze(1) # shape (seq_length, 1) i.e. makes it vertical
        
        div_term = torch.exp(
            (torch.arange(0, d_model, 2)/d_model)*(-4)*log(10)
        ).to(dev)
        
        pe = torch.zeros(size=(seq_length, d_model), dtype=torch.float32, device=dev) # the tensor to be multiplied to positions tensor to get pe
        pe[:, 0::2] = torch.sin(positions*div_term) # for even dimensions
        pe[:, 1::2] = torch.cos(positions*div_term) # for odd dimensions
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1) # copy out the encodings for each batch
        return pe
    
    def forward(self, x):
        batch_size, seq_length = x.shape

        # step 1: make embeddings
        token_embedding = self.embedding(x)

        # step 2: go from d_embed to d_model
        token_embedding = self.projection(token_embedding) \
            * self.scaling # multiplying with scaling factor, just like in the paper

        # step 3: add positional encoding
        pos_encoding = self.create_positional_encoding(
            seq_length=seq_length, 
            d_model = self.d_model,
            batch_size=batch_size,
            dev=self.dev
        )

        #step 4: normalize the sum of pos encoding and token_embed
        norm_sum = self.layerNorm(pos_encoding + token_embedding)
        op = self.dropout(norm_sum)
        return op

class TransformerAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 dropout_p: float = 0.1,
                 dev='cpu'
                 ):
        super().__init__()
        if (d_model % num_heads) != 0: raise ValueError(f'`d_model` not divisible by `num_heads`')
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = self.d_model // self.num_heads
        self.scale_factor = float(1.0 / sqrt(self.d_heads))
        self.dropout = nn.Dropout(p=dropout_p)
        self.dev = dev

        #linear transformations
        self.q_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.dev
        )

        self.k_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.dev
        )

        self.v_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.dev
        )

        self.output_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            device=self.dev
        )

    def forward(self, 
                seq: torch.Tensor, 
                key_value_states:torch.Tensor = None, 
                att_mask: torch.Tensor = None):
        batch_size, seq_length, d_model = seq.size()

        Q_state: torch.Tensor = self.q_proj(seq)
        if key_value_states is not None:
            kv_seq_len = key_value_states.size(1)
            K_state: torch.Tensor = self.k_proj(key_value_states)
            V_state: torch.Tensor = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_length
            K_state: torch.Tensor = self.k_proj(seq)
            V_state: torch.Tensor = self.v_proj(seq)

        Q_state = Q_state.view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)
        K_state = K_state.view(batch_size, kv_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_heads, self.d_heads).transpose(1, 2)

        Q_state = Q_state * self.scale_factor
        
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1, -2))
        

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask # yes, in this case the mask is not multiplied, but added. This is to ensure that after softmax the things to be excluded are 0
        
        att_score = F.softmax(self.att_matrix, dim=-1) # torch.nn.Softmax() is used in __init__, F.softmax() is used for these inline operations.
        att_score = self.dropout(att_score)
        att_op = torch.matmul(att_score, V_state)

        #concatenating all heads 
        att_op = att_op.transpose(1, 2)
        att_op = att_op.contiguous().view(batch_size, seq_length, self.num_heads*self.d_heads)

        att_op = self.output_proj(att_op)

        return att_op

class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dev='cpu'):
        
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dev = dev

        self.fc1 = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
            device=self.dev
        )

        self.fc2 = nn.Linear(
            in_features=self.d_ff,
            out_features=self.d_model,
            device=self.dev
        )
        
    def forward(self, input:torch.Tensor):
        batch_size, seq_length, d_input = input.size()
        f1 = F.relu(self.fc1(input))
        f2 = self.fc2(f1)
        return f2

class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 dropout_p = 0.1,
                 dev='cpu'
                 ):
        
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.d_ff = d_ff
        self.dev = dev

        self.att_layer = TransformerAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p,
            dev=self.dev
        )

        self.ffn = FeedForwardNetwork(
            d_model=self.d_model,
            d_ff = self.d_ff,
            dev = self.dev
        )

        self.norm1 = nn.LayerNorm(self.d_model, device=self.dev)
        self.norm2 = nn.LayerNorm(self.d_model, device=self.dev)

        self.dropout = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x: torch.Tensor):
        x_att = self.att_layer(x)

        x_att = self.dropout(x_att)
        x_norm1 = self.norm1(x + x_att)

        x_ff = self.ffn(x_norm1)

        x_ff = self.dropout(x_ff)
        x_norm2 = self.norm2(x_ff + x_norm1)
        
        return x_norm2
    
class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 dropout_p = 0.1,
                 dev='cpu'
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.dev = dev

        self.att_layer1 = TransformerAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p,
            dev=self.dev
        )

        self.norm1 = nn.LayerNorm(self.d_model, device=self.dev)

        self.att_layer2 = TransformerAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p,
            dev=self.dev
        )

        self.norm2 = nn.LayerNorm(self.d_model, device=self.dev)

        self.ffn = FeedForwardNetwork(
            d_model=self.d_model,
            d_ff = self.d_ff,
            dev=self.dev
        )

        self.norm3 = nn.LayerNorm(self.d_model, device=self.dev)

        self.dropout = nn.Dropout(p=self.dropout_p)

    @staticmethod
    def create_causal_mask(seq_len: int, dev='cpu') -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=dev), diagonal=1)
        mask = mask.masked_fill(mask == 1, value=float('-inf'))
        return mask

    def forward(self, x: torch.Tensor, 
                cross_input:torch.Tensor,
                padding_mask:torch.Tensor = None
                ):
        batch_size, seq_length, d_model = x.size()

        causal_mask = self.create_causal_mask(seq_len=seq_length, dev=self.dev)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) #unsqeeze the mast for self attention

        x_att1 = self.att_layer1( #self-attention
            seq=x,
            att_mask = causal_mask
        )

        x_att1 = self.dropout(x_att1)
        x_norm1 = self.norm1(x_att1 + x)

        x_att2 = self.att_layer2( #cross attention
            seq=x_norm1,
            key_value_states=cross_input,
            att_mask = padding_mask
        )

        x_att2 = self.dropout(x_att2 + x_norm1)
        x_norm2 = self.norm2(x_att2)

        x_ff = self.ffn(x_norm2)

        x_ff = self.dropout(x_ff)
        x_norm3 = self.norm3(x_ff)
        
        return x_norm3


class TransformerEncoderDecoder(nn.Module):
    def __init__(self,
                 N_enc: int,
                 N_dec: int, 
                 d_model:int,
                 num_heads: int, 
                 d_ff: int,
                 dropout_p = 0.1,
                 dev = 'cpu'
                 ):
        
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.dev = dev

        self.encoder_stack = nn.ModuleList([
            TransformerEncoder(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_p=self.dropout_p,
                dev=self.dev
            ) for _ in range(N_enc)
        ])

        self.decoder_stack = nn.ModuleList([
            TransformerDecoder(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_p=self.dropout_p,
                dev = self.dev
            ) for _ in range(N_dec)
        ])

    def forward(self, x: torch.Tensor, y:torch.Tensor, padding_mask=None) -> torch.Tensor:
        #pass through the encoder stack
        encoder_output = x
        for encoder in self.encoder_stack:
            encoder_output = encoder(encoder_output)

        #pass through the decoder stack
        #uses only the final encoder input
        decoder_output = y
        for decoder in self.decoder_stack:
            decoder_output = decoder(decoder_output, cross_input=encoder_output)

        return decoder_output
    

class Transformer(nn.Module):
    def __init__(self,
                 N_enc: int,
                 N_dec: int,
                 vocab_size:int, 
                 d_embed: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 d_tgt_vocab: int,
                 dropout_p = 0.1,
                 dev='cpu'
                 ):
        super().__init__()

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.d_tgt_vocab = d_tgt_vocab
        self.dev= dev

        self.src_embedder = EmbeddingWithPositionalEncoding(
            vocab_size=self.vocab_size,
            d_embed=self.d_embed,
            d_model=self.d_model,
            dropout_p=self.dropout_p,
            dev=self.dev
        )

        self.tgt_embedder = EmbeddingWithPositionalEncoding(
            vocab_size=self.vocab_size,
            d_embed=self.d_embed,
            d_model=self.d_model,
            dropout_p=self.dropout_p,
            dev = self.dev
        )

        self.encoder_decoder_stack = TransformerEncoderDecoder(
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_p=self.dropout_p,
            dev = self.dev
        )

        self.output_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_tgt_vocab,
            device=self.dev
        )

        self.softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def shift_target_right(tgt_tokens: torch.Tensor, dev='cpu') -> torch.Tensor:
        batch_size, seq_len = tgt_tokens.size() # no d_model since, no Embedding done
        zer = torch.zeros(
            size=(batch_size, 1),
            device=dev,
            dtype=torch.long
        )
        return torch.concat([
            zer, 
            tgt_tokens[:, :-1]], 
            dim=1).to(dev)

    def forward(self, 
                src_tokens:torch.Tensor, 
                tgt_tokens:torch.Tensor,
                padding_mask=None
                ) -> torch.Tensor:
        
        tgt_tokens = self.shift_target_right(tgt_tokens, dev=self.dev) 
        # shifting is needed to prevent information leakage. 
        # it allows parallel traning in spite of hiding the token 
        # to be predicted.
        inp_embed = self.src_embedder(src_tokens)
        tgt_embed = self.tgt_embedder(tgt_tokens)

        enc_dec_out = self.encoder_decoder_stack.forward(inp_embed, tgt_embed, padding_mask=padding_mask)
        
        out = self.output_proj(enc_dec_out)
        log_probs = self.softmax(out)
        return log_probs
