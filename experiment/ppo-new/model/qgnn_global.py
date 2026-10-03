"""Local QGNN features fused with per-circuit global context."""
import torch
from torch import nn
from model.qgnn import QGNN

class QGNNGlobal(QGNN):
    def __init__(self, num_layers, num_gate_types, gate_type_embed_dim, h_feats, inter_dim):
        super().__init__(num_layers, num_gate_types, gate_type_embed_dim, h_feats, inter_dim)
        self.context_fusion = nn.Sequential(nn.Linear(2 * h_feats + 1, h_feats), nn.ReLU())
        nn.init.xavier_uniform_(self.context_fusion[0].weight)
        nn.init.zeros_(self.context_fusion[0].bias)

    def forward(self, g):
        local = super().forward(g)
        # DGL batch boundaries isolate circuits; never pool across the batch.
        counts_cpu = g.batch_num_nodes()
        counts = counts_cpu.to(device=local.device)
        graph_ids = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=local.device), counts,
            output_size=local.shape[0],
        )
        pooled = local.new_zeros((counts.shape[0], local.shape[1]))
        pooled = pooled.index_add(0, graph_ids, local)
        pooled = pooled / counts.to(local.dtype).clamp_min(1).unsqueeze(1)
        context = pooled[graph_ids]
        size = torch.log1p(counts.to(local.dtype))[graph_ids].unsqueeze(1)
        return self.context_fusion(torch.cat([local, context, size], dim=1))
