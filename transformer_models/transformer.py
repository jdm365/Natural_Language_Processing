import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class PositionalEncoder:
    def __init__(self, input_dims, embedding_dims):
         self.input_dims = input_dims
         self.embedding = np.linspace(0, embedding_dims)
         self.embedding_dims = embedding_dims

        def encode(self, input_string):
            ## returns np.array with dims (batch_size, input_dims, embedding_dims)
            encodings = []
            for i in range(len(input_string)):
                denominator = (i / 10000) ** (2*self.embedding/self.embedding_dims))
                encodings.append(i / denominator)
            return np.stack(encodings)


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads):
        super(MultiHeadedAttention, self).__init__()
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.norm_factor = np.sqrt(embedding_dims)
        self.qkv_dims = embedding_dims // n_heads
        self.encoder = PositionalEncoder(input_dims, embedding_dims)
        self.n_heads = n_heads

        self.q_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)
        self.k_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)
        self.v_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)

        self.output_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)

    def forward(self, inputs):
        encoded_vectors = self.encoder.forward(inputs)
        
        batch_size = encoded_vectors[0]
        new_dims = (batch_size, self.input_dims, self.n_heads, self.qkv_dims)

        queries = self.q_proj(encoded_vectors)
        keys = self.k_proj(encoded_vectors)
        values = self.v_proj(encoded_vectors)
        
        queries = queries.reshape(*new_dims)
        keys = keys.reshape(*new_dims)
        values = values.reshape(*new_dims)
        ## dims (batch_size, n_heads, input_dims, qkv_dims)

        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        ## dims (batch_size, n_heads, input_dims, qkv_dims)

        out = T.einsum('stuv, stvw -> stuw', queries, \
                keys.transpose(-2, -1)) / self.norm_factor
        out = F.softmax(out, dim=-1)
        ## dims (batch_size, n_heads, input_dims, input_dims)

        attention_values = T.einsum('stuu, stuv -> stuv', \
                out, values).permute(0, 2, 1, 3).contiguous()
        ## dims (batch_size, input_dims, n_heads, qkv_dims)

        attention_values = attention_values.flatten(start_dim=2)
        ## dims (batch_size, input_dims, embedding_dims)

        return self.output_proj(attention_values) + encoded_vectors



class Encoder(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads):
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.n_heads = n_heads
        
        self.mha = MultiHeadedAttention(input_dims, embedding_dims, n_heads)
        self.norm_1 = nn.LayerNorm(emedding_dims)

        self.mlp = nn.Sequential(
                nn.Linear(embedding_dims, 4 * embedding_dims),
                nn.LeakyReLU(),
                nn.Linear(4 * embedding_dims, embedding_dims)
                )
        self.norm_2 = nn.LayerNorm(embedding_dims)


    def forward(self, inputs):
        mha_output = self.norm_1(self.mha.forward(inputs))
        mlp_output = self.norm_2(self.mlp(mha_output) + mha_output)
        return mlp_output
        


class EncoderBlock(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads, n_blocks=2):
        self.encoder_block = nn.ModuleList(
                [Encoder(input_dims, embedding_dims, n_heads) for _ in range(n_blocks)])
        

## TODO: Make decoder and create general transformer (decoder optional)
##       Look into adding CLS token
































