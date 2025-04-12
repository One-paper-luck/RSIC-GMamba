from dataclasses import dataclass, fields, asdict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.transformer.mamba_decoder import RMSNorm, Mamba, MambaConfig

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    # pscan: bool = False
    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)



class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig,vocab_size, max_len, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                     dropout=.1,self_att_module=None, enc_att_module=None, self_att_module_kwargs=None,
                     enc_att_module_kwargs=None):
        super().__init__()
        self.d_model=d_model
        lm_config.d_model = d_model
        lm_config.vocab_size = vocab_size
        lm_config.pscan = False
        self.lm_config = lm_config
        self.config = lm_config().to_mamba_config()
        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model, padding_idx=padding_idx)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.fc = nn.Linear(self.config.d_model * 2, self.config.d_model, bias=False)
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tokens, EncoderOut=None,mask_enc=None):
        # tokens : (B, L)
        # logits : (B, L, vocab_size)
        # EncoderOut
        x = self.embedding(tokens)
        enc1=EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        enc2=EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        enc3=EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([x, enc1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([x, enc2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([x, enc3], -1)))
        enc_mesh = (enc1 * alpha1 + enc2 * alpha2 + enc3 * alpha3) / np.sqrt(3)
        enc_mesh = self.mamba(enc_mesh)
        enc_mesh = self.norm_f(enc_mesh)
        logits = self.lm_head(enc_mesh)

        return F.log_softmax(logits, dim=-1)

    def step(self, token, caches, EncoderOut):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        enc1 = EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        enc2 = EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        enc3 = EncoderOut[:, 0].mean(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([x, enc1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([x, enc2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([x, enc3], -1)))
        enc_mesh = (enc1 * alpha1 + enc2 * alpha2 + enc3 * alpha3) / np.sqrt(3)

        x=enc_mesh.squeeze(1)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)
        logits=F.log_softmax(logits, dim=-1)

        return logits, caches


    def generate(self, images, tokenizer, num_tokens: int = 20, batch_size: int = 1, sample: bool = True,
                 top_k: int = 5, temperature: float = 1.0):
        # self.eval()
        input_ids = torch.tensor(tokenizer.vocab.stoi['<bos>']).cuda()
        input_ids = input_ids.repeat(batch_size, 1)

        seq_logprob = torch.zeros(batch_size, 1).cuda()

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv - 1, device=input_ids.device))
                  for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):

            # with torch.no_grad():
                # forward the new output, get new cache
            next_token_logits, caches = self.step(input_ids[:, i], caches,images)  # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i + 1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1)  # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k)  # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1)  # (batch_size)

                next_token_probs = torch.gather(probs, dim=1, index=next_token.unsqueeze(1))
                seq_logprob=torch.cat([seq_logprob, next_token_probs], dim=1)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        outputs = input_ids[:, 1:]
        # outputs = [tokenizer.decode(output.tolist()) for output in input_ids]
        outputs_seq_logprob=seq_logprob[:,1:]

        if batch_size == 1:
            return outputs[0],outputs_seq_logprob[0]
        else:
            return outputs, outputs_seq_logprob


