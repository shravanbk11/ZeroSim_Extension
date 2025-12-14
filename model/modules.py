import math
import torch
import torch.nn as nn
from torch.nn.modules import transformer
from functools import partial


class CircuitTranformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_type: str = "layernorm",
        bias: bool = False,
        cross_attn: bool = False,
        activation: str = "gelu",
        ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

        self.cross_attn_flag = cross_attn
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            create_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias),
            nn.Dropout(dropout),
        )
        if norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm1d(d_model, eps=layer_norm_eps, bias=bias)
            self.norm2 = nn.BatchNorm1d(d_model, eps=layer_norm_eps, bias=bias)
        elif norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x, 
        cross=None,
        key_padding_mask=None, 
        attn_mask=None, 
        cross_key_padding_mask=None,
        cross_attn_mask=None, 
        attn_bias=None,
        layer = None
    ):
        residual = x

        x, self_attn_weights = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        # cross attention with the graph representation
        if self.cross_attn_flag and cross is not None:
            residual = x
            x, cross_attn_weights = self.cross_attn(
                x, cross, cross, key_padding_mask=cross_key_padding_mask, attn_mask=cross_attn_mask)
            x = self.dropout(x)
            x = self.norm1(x + residual)

        # feedforward
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return x
    

class NodeEmbedding(nn.Module):
    def __init__(self, num_device_types: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_device_types, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return math.sqrt(self.embed_dim) * self.embedding(x)
    

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
    

class MLP_decoder(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)
