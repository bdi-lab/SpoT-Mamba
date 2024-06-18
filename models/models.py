import torch
import torch.nn as nn
import dgl

from typing import  Tuple, Optional

from models.layers import WalkEmbedding, TemporalEmbedding, MambaBlock

P_BFS = 0.5 # < 1
Q_BFS = 2. # > 1
P_DFS = 2. # > 1
Q_DFS = 0.5 # < 1
P_RW = 1.
Q_RW = 1.

class SpoTMamba(nn.Module):
    
    def __init__(self, in_dim: int,
                 out_dim: int,
                 emb_dim: int,
                 ff_dim: int,
                 graph: dgl.DGLGraph,
                 steps_per_day: int,
                 seed: int,
                 
                 p_bfs=P_BFS,
                 q_bfs=Q_BFS,
                 p_dfs=P_DFS,
                 q_dfs=Q_DFS,
                 p_rw=P_RW,
                 q_rw=Q_RW,
                 num_walks: int=5,
                 len_walk: int=10,
                 
                 num_layers: int=4,
                 dropout: float=0.1) -> None:
        
        super().__init__()
        
        # Model Configurations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        
        self.graph = graph
        self.steps_per_day = steps_per_day
        self.p_bfs = p_bfs
        self.q_bfs = q_bfs
        self.p_dfs = p_dfs
        self.q_dfs = q_dfs
        self.p_rw = p_rw
        self.q_rw = q_rw
        self.num_walks = num_walks
        self.total_walks = 3*num_walks
        self.len_walk = len_walk
        self.num_layers = num_layers
        
        # Walk Sequences
        sequence, cost = self.generate_walk_sequences(seed) # (num_nodes, len_walk, total_walks)
        self.sequence = sequence
        self.cost = cost
        
        # Input Embedding
        self.walk_emb = WalkEmbedding(emb_dim, graph.num_nodes(), num_walks, len_walk)
        self.temp_emb = TemporalEmbedding(in_dim, emb_dim, steps_per_day)
        
        # Blocks
        emb_walk_dim = self.total_walks * (3*emb_dim)
        emb_scan_dim = 4*emb_dim
        self.emb_walk_dim = emb_walk_dim
        self.emb_scan_dim = emb_scan_dim
        
        self.walk_blocks = nn.ModuleList([
            MambaBlock(emb_walk_dim,
                    layer_idx=l,
                    bi_directional=True)
            for l in range(num_layers)])
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_scan_dim, nhead=4, dim_feedforward=ff_dim, dropout=dropout),
            num_layers
        )
        
        self.temporal_blocks = nn.ModuleList([
            MambaBlock(emb_scan_dim,
                    layer_idx=l+num_layers)
            for l in range(num_layers)])
        
        # Walk Sequence Projection
        self.walk_conv = nn.Conv1d(len_walk, 1, kernel_size=1)
        self.ff_walk = nn.Sequential(
            nn.Linear(emb_walk_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        # Spatio-Temporal Projection
        self.ff_scan = nn.Sequential(
            nn.Linear(emb_scan_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        # Regression Layer
        self.regression_layer = nn.Sequential(
            nn.Linear(self.emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.out_dim)
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor, out_steps: int=12) -> torch.Tensor:
        
        # x: (batch_size, in_steps (temporal), num_nodes (spatial), in_dim(=3))
        ## last dimesion -> [traffic flow, timestamp of day, day of week]
        # out: (batch_size, out_steps, num_nodes, out_dim(=1))
        
        device = x.device
        
        # Walk Sequence Embedding
        sequence = self.sequence.to(device)
        cost = self.cost.to(device)
        h_walk = self.walk_emb(sequence, cost, self.graph)
        
        h_walk = h_walk.view(h_walk.shape[0], h_walk.shape[1], -1) # (num_nodes, len_walk, total_walks * 3*emb_dim)
        residual = None
        for block in self.walk_blocks:
            h_walk, residual = block(h_walk, residual)
        
        h_walk = self.activation(h_walk)
        h_walk = self.walk_conv(h_walk).squeeze(1) # (num_nodes, total_walks * 3*emb_dim)
        h_walk = self.ff_walk(h_walk) # (num_nodes, emb_dim)
    
        # Spatio-Temporal Scan
        out = self.predict_next_step(x, h_walk)

        return out

    def generate_walk_sequences(self, seed) -> Tuple[torch.Tensor, torch.Tensor]:
        
        dgl.seed(seed)
        
        graph = self.graph
        num_walks = self.num_walks
        num_edges = self.len_walk-1
        nodes = graph.nodes().repeat_interleave(num_walks)
        
        # Generate Walk Sequences
        sequence_bfs, eids_bfs = dgl.sampling.node2vec_random_walk(graph, nodes, self.p_bfs, self.q_bfs, num_edges, return_eids=True)
        sequence_dfs, eids_dfs = dgl.sampling.node2vec_random_walk(graph, nodes, self.p_dfs, self.q_dfs, num_edges, return_eids=True)
        sequence_rw, eids_rw = dgl.sampling.node2vec_random_walk(graph, nodes, self.p_rw, self.q_rw, num_edges, return_eids=True)
        
        # Node ID Sequence
        sequence = torch.cat([sequence_bfs.view(-1, num_walks, self.len_walk),
                              sequence_dfs.view(-1, num_walks, self.len_walk),
                              sequence_rw.view(-1, num_walks, self.len_walk)],
                             dim=1) # (num_nodes, total_walks, len_walk)
        sequence = sequence.permute(0, 2, 1)  # (num_nodes, len_walk, total_walks)
        
        # Cost Sequence
        cost_bfs = graph.edata['cost'][eids_bfs]
        cost_dfs = graph.edata['cost'][eids_dfs]
        cost_rw = graph.edata['cost'][eids_rw]
        cost = torch.cat([cost_bfs.view(-1, num_walks, num_edges),
                          cost_dfs.view(-1, num_walks, num_edges),
                          cost_rw.view(-1, num_walks, num_edges)],
                         dim=1) # (num_nodes, 3*num_walks, len_walk-1)
        zeros = torch.zeros(cost.shape[0], cost.shape[1], 1)
        cost = torch.cat([zeros, cost], dim=-1) # (num_nodes, 3*num_walks, len_walk)
        cost = cost.permute(0, 2, 1) # (num_nodes, total_walks, len_walk)
        
        return sequence, cost

    def predict_next_step(self, x: torch.Tensor,
                          h_walk: Optional[torch.Tensor]) -> torch.Tensor:
        '''
            Conduct temporal scan & spatial mixing and predict the next step of the time series
            
            - x: (batch_size, T, num_nodes, in_dim) ~ 0:T
            - h_walk: (num_nodes, emb_dim)
            - out: (batch_size, T, num_nodes, out_dim) ~ T+1:T+T
        '''
        
        batch_size, num_steps, num_nodes, _ = x.shape
        
        # Input Processing
        temp_emb = self.temp_emb(x)
        h_walk = h_walk.expand(batch_size, num_steps, *h_walk.shape)
        h = torch.cat([temp_emb, h_walk], dim=-1) # (batch_size, T, num_nodes, emb_scan_dim)
        
        # Temporal Scan
        h_temporal = h.transpose(1, 2) # (batch_size, num_nodes, T, emb_scan_dim)
        h_temporal = h_temporal.reshape(batch_size*num_nodes, num_steps, -1)
        
        residual = None
        for block in self.temporal_blocks:
            h_temporal, residual = block(h_temporal, residual)
        h_temporal = h_temporal.reshape(batch_size, num_nodes, num_steps, -1)
        h_temporal = h_temporal.transpose(1, 2) # (batch_size, T, num_nodes, emb_scan_dim)
        
        # Spatial Mixing
        h_spatial = h_temporal.reshape(batch_size*num_steps, num_nodes, -1)
        h_spatial = self.transformer_encoder(h_spatial)
        h_spatial = h_spatial.view(batch_size, num_steps, num_nodes, -1)
        h = self.activation(h_spatial)
        h = self.ff_scan(h)
        h = self.activation(h)
        
        # Regression Layer
        out = self.regression_layer(h)
        
        return out
