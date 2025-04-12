from torch.nn import functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
from models.containers import Module

from models.transformer.mamba_lm import MambaLM,MambaLMConfig


class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        MambaLMConfig.d_model=d_model
        MambaLMConfig.vocab_size=vocab_size
        MambaLMConfig.pscan=False
        self.d_model=d_model
        self.MambaDecoder=MambaLM(MambaLMConfig)
    def forward(self, input, encoder_output, mask_encoder):
        out=self.MambaDecoder(input,encoder_output)
        return F.log_softmax(out, dim=-1)

