from torch.nn import functional as F
import torch
from torch import nn
from models.transformer.mamba_encoder import Mamba, MambaConfig, RMSNorm


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        config = MambaConfig()
        self.MambaEncoder = Mamba(config)


    def forward(self, input, attention_weights=None):
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        outs = self.MambaEncoder(input)
        return outs, attention_mask


class VisualEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=768, **kwargs):
        super(VisualEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(VisualEncoder, self).forward(out, attention_weights=attention_weights)
