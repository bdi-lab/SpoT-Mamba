import torch
import torch.nn as nn
import dgl
from mamba_ssm import Mamba

from typing import Tuple, Optional

################################# SpoTMamba #################################
class WalkEmbedding(nn.Module):
    
    def __init__(self,
                 emb_dim: int,
                 num_nodes: int,
                 num_walks: int=4,
                 len_walk: int=8) -> None:
        
        super().__init__()
        self.emb_dim = emb_dim
        self.num_walks = num_walks
        self.len_walk = len_walk
        
        self.cost_embedding = nn.Linear(1, emb_dim)
        self.degree_embedding = nn.Linear(1, emb_dim)
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        
    def forward(self, sequence: torch.Tensor,
                cost: torch.Tensor,
                graph: dgl.DGLGraph) -> torch.Tensor:
        
        # sequence: (num_nodes, len_walk, total_walks)
        # cost: (num_nodes, len_walk, total_walks)
        # out: (num_nodes, len_walk, total_walks 3*emb_dim)
        
        device = sequence.device
        
        # Degree Embedding
        degrees = graph.in_degrees().unsqueeze(-1).float().to(device) # (num_nodes, 1)
        deg_emb = self.degree_embedding(degrees) # (num_nodes, emb_dim)
        deg_emb = deg_emb[sequence] # (num_nodes, len_walk, total_walks, emb_dim)
        
        # Cost Embedding
        cost = cost.unsqueeze(-1).float().to(device)
        cost_emb = self.cost_embedding(cost) # (num_nodes, len_walk, total_walks, emb_dim)
        
        # Node Embedding
        node_emb = self.node_emb(sequence) # (num_nodes, len_walk, total_walk, emb_dim)
        
        # Concatenation
        out = torch.cat([deg_emb, cost_emb, node_emb], dim=-1) # (num_nodes, len_walk, total_walks, 3*emb_dim)

        return out

class TemporalEmbedding(nn.Module):
    
    def __init__(self, in_dim: int,
                 emb_dim: int,
                 steps_per_day: int) -> None:
        
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.steps_per_day = steps_per_day
        
        self.feat_embedding = nn.Linear(in_dim, emb_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, emb_dim) # Timestamp-of-Day embedding
        self.dow_embedding = nn.Embedding(7, emb_dim) # Day-of-Week embedding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (batch_size, in_steps, num_nodes, in_dim)
        # out: (batch_size, in_steps, num_nodes, 3*emb_dim)
        
        feat = x[..., :self.in_dim]
        tod = x[..., 1]
        dow = x[..., 2]
        out = []
        
        # Feature Embedding
        feat_emb = self.feat_embedding(feat)
        out.append(feat_emb)
        
        # Time-of-Day Embedding
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        out.append(tod_emb)
        
        # Day-of-Week Embedding
        dow_emb = self.dow_embedding(dow.long())
        out.append(dow_emb)
        
        # Concatenation
        out = torch.cat(out, dim=-1)
        
        return out

class MambaBlock(nn.Module):
    
    def __init__(self, dim: int,
                 norm_cls=nn.LayerNorm,
                 layer_idx: Optional[int]=None,
                 bi_directional: bool=False) -> None:
        """
        From "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py"
        
        Simple block wrapping a Mamba with LayerNorm/RMSNorm and residual connection

        Here we have: Add -> LN -> Mamba, returning both
        the hidden_states (output of the Mamba) and the residual.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.dim = dim
        self.bi_directional = bi_directional
        
        self.norm = norm_cls(dim)
        self.mamba = Mamba(dim, layer_idx=layer_idx) # TODO: Setting Hyperparameters
        if bi_directional:
            self.mamba_inverse = Mamba(dim, layer_idx=-layer_idx)
            self.linear = nn.Linear(dim, dim)
        
    def forward(self, hidden_states: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                inference_params=None):
        
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mamba(LN(residual))
        """
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        h_norm = self.norm(residual.to(dtype=self.norm.weight.dtype))
        
        hidden_states = self.mamba(h_norm, inference_params=inference_params)
        if self.bi_directional:
            h_inverse = self.mamba_inverse(torch.flip(h_norm, [-2]),
                                           inference_params=inference_params)
            hidden_states = hidden_states + torch.flip(h_inverse, [-2])
            hidden_states = self.linear(hidden_states)
        
        return hidden_states, residual


#############################################################################
