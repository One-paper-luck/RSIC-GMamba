import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.pscan import pscan
from einops import rearrange
import random
import numpy as np
"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int = 512  # D
    n_layers: int = 3
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    res_drop = 0.1

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class TokenReplacement:
    @staticmethod
    def replace_tokens(tensor, ratio):
        padding_length = int(tensor.size(1) * ratio)
        replace_indices = torch.randperm(tensor.size(1))[:padding_length]

        replaced_tensor = torch.zeros_like(tensor)
        replaced_tensor[:, :tensor.size(1) - padding_length, :] = tensor[:, replace_indices, :]

        return replaced_tensor

    # 将 patch 张量分组
    @staticmethod
    def group_patches(patch_tensor, num_groups):
        batch_size, patch_count, patch_dim = patch_tensor.size()
        patches_per_group = (patch_count + num_groups - 1) // num_groups  # 向上取整计算每组的 patch 数量
        padding_size = patches_per_group * num_groups - patch_count  # 计算需要填充的数量
        pad = torch.zeros(batch_size, padding_size, patch_dim).cuda()  # 创建填充张量
        padded_patch_tensor = torch.cat((patch_tensor, pad), dim=1)  # 填充原始张量
        grouped_patches = padded_patch_tensor.view(batch_size, num_groups, patches_per_group, patch_dim)
        return grouped_patches




    # 变异操作 """每个batch的随机索引一样"""
    @staticmethod
    def mutate_feature(image_feature, use_random_normal=False, mutation_rate=0.1):

        device = image_feature.device
        batch_size, num_feature, feature_dim = image_feature.size()

        # 计算需要变异的向量数量
        num_mutations = int(mutation_rate * num_feature)
        # 生成要变异的向量的索引
        mutation_indices = torch.randperm(num_feature)[:num_mutations].to(device)
        # 生成随机索引张量
        replacement_indices = torch.randperm(num_feature)[:num_mutations].to(device)
        mutated_features = image_feature.clone()
        # 扩展索引张量的维度，并复制batch_size次以适配特征张量的形状
        expanded_mutation_indices = mutation_indices.unsqueeze(0).unsqueeze(2).expand(batch_size, num_mutations,
                                                                                      feature_dim)
        expanded_replacement_indices = replacement_indices.unsqueeze(0).unsqueeze(2).expand(batch_size,
                                                                                            num_mutations,
                                                                                            feature_dim)

        # 根据扩展后的索引张量进行scatter_操作
        if use_random_normal:
            random_normal = torch.randn(batch_size, num_mutations, feature_dim).to(device)
            mutated_features.scatter_(1, expanded_mutation_indices, random_normal)
        else:
            mutated_features.scatter_(1, expanded_mutation_indices,image_feature.gather(1, expanded_replacement_indices))

        return mutated_features


    # 变异操作 每个batch的随机索引不一样"""
    @staticmethod
    def mutate_feature_dif(image_feature, use_random_normal=True, mutation_rate=0.1):
        device = image_feature.device
        batch_size, num_feature, feature_dim = image_feature.size()

        # 计算需要变异的向量数量
        num_mutations = int(mutation_rate * num_feature)

        # 生成要变异的向量的索引
        mutation_indices = torch.zeros(batch_size, num_mutations, dtype=torch.long).to(device)
        for i in range(batch_size):
            mutation_indices[i] = torch.randperm(num_feature)[:num_mutations]

        # 生成随机索引张量
        replacement_indices = torch.zeros(batch_size, num_mutations, dtype=torch.long).to(device)
        for i in range(batch_size):
            replacement_indices[i] = torch.randperm(num_feature)[:num_mutations]

        mutated_features = image_feature.clone()
        for i in range(batch_size):
            if use_random_normal:
                random_vector = torch.randn(num_mutations, feature_dim).to(device)
                mutated_features[i][mutation_indices[i]] = random_vector
            else:
                mutated_features[i].scatter_(0, mutation_indices[i].unsqueeze(1).expand(num_mutations, feature_dim),
                                             image_feature[i, replacement_indices[i]])

        return mutated_features

    @staticmethod
    def create_individual(image_feature, num_individual=4): # 4 8
        batch_size, num_feature, dim = image_feature.shape
        num_sub_individual = num_feature // num_individual
        if num_feature % num_individual != 0:
            padding_length = (num_sub_individual + 1) * num_individual - num_feature
            pad_tensor = torch.zeros(batch_size, padding_length, dim, device=image_feature.device)
            image_feature = torch.cat((image_feature, pad_tensor), dim=1)
        else:
            padding_length = None

        individuals = rearrange(image_feature, "b (n k) d -> b n k d", n=num_individual)  #
        return individuals, padding_length

    @staticmethod
    def crossover(individuals, k=2):
        batch_size, num_individual, num_sub_individual, dim = individuals.shape
        # 创建用于记录变异操作的组索引的张量
        crossed_indices = []

        # 随机选择两个组进行交叉操作
        for i in range(batch_size):
            # 随机选择两个不同的组索引
            group_indices = torch.randperm(num_individual)[:2]
            # 执行交叉操作
            temp = individuals[i, group_indices[0], -k:].clone()
            individuals[i, group_indices[0], -k:] = individuals[i, group_indices[1], -k:]
            individuals[i, group_indices[1], -k:] = temp

            # 记录发生交叉操作的组索引
            crossed_indices.append((group_indices[0].item(), group_indices[1].item()))

        individuals = rearrange(individuals, "b n k d -> b (n k) d", n=num_individual)
        return individuals, crossed_indices

    @staticmethod
    def reverse_crossover(individuals, crossed_indices, num_individual=4,k=2):
        batch_size, _, _ = individuals.shape
        individuals = rearrange(individuals, "b (n k) d -> b n k d", n=num_individual)
        for i in range(batch_size):
            group_indices = crossed_indices[i]
            # 执行交叉操作
            temp = individuals[i, group_indices[0], -k:].clone()
            individuals[i, group_indices[0], -k:] = individuals[i, group_indices[1], -k:]
            individuals[i, group_indices[1], -k:] = temp

        individuals = rearrange(individuals, "b n k d -> b (n k) d", n=num_individual)
        return individuals

    @staticmethod
    def remove_padding(x, padding_length):
        if padding_length is not None:
            x = x[:, :-padding_length, :]

        return x



class MutiHeadSelfAttention(nn.Module):
    '''
    Muti-Head Self-Attention
    '''
    def __init__(self, dim, attention_dim, d_state, h=1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MutiHeadSelfAttention, self).__init__()
        attention_dim = attention_dim//16
        self.decoder_att = nn.Linear(dim * d_state, attention_dim)
        d_k = attention_dim // h
        d_v = d_k
        self.fc_q = nn.Linear(attention_dim, h * d_k)
        self.fc_k = nn.Linear(attention_dim, h * d_k)
        self.fc_v = nn.Linear(attention_dim, h * d_v)
        self.fc_o = nn.Linear(h * d_v, dim * d_state)
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, hidden):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq, decoder_dim, state = hidden.shape

        queries = self.decoder_att(hidden.contiguous().view(b_s, nq, -1))
        keys = queries
        values = queries
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out).view(b_s, nq, decoder_dim, state)  # (b_s, nq, d_model)

        return out



class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.layers = nn.Sequential(ResidualBlock(config,dilation=2),
                                    ResidualBlock(config,dilation=4),
                                    ResidualBlock(config, dilation=8))

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x.unsqueeze(1))

        outs = torch.cat(outs, 1)

        return outs



class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig,dilation=None):
        super().__init__()
        self.mixer = MambaBlock(config,dilation=dilation)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.mlp_channels = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.res_drop)
        )
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        output = output + self.mlp_channels(self.norm2(output))

        return output



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig,dilation=None):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=dilation,
                      dilation=dilation), nn.BatchNorm2d(512, affine=False), nn.ReLU(inplace=True))

        self.attention = MutiHeadSelfAttention(dim=config.expand_factor * config.d_model, attention_dim=config.d_model,
                                               d_state=config.d_state)
        self.attention_b = MutiHeadSelfAttention(dim=config.expand_factor * config.d_model, attention_dim=config.d_model,
                                               d_state=config.d_state)


        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.A_log_b = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log_b._no_weight_decay = True

        self.D_b = nn.Parameter(torch.ones(config.d_inner))
        self.D_b._no_weight_decay = True



        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False


    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        # Genetic SSM
        # dilation
        x_d = x.permute(0, 2, 1)
        x_d = x_d.reshape(x_d.shape[0], x_d.shape[1], 7, 7)
        x_d = self.net(x_d)
        x = x_d.reshape(x_d.shape[0], x_d.shape[1], x_d.shape[2] * x_d.shape[3]).permute(0, 2, 1) + x

        _, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        x_b=x

        # x branch  crossover
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)
        x = F.silu(x)

        crossover_flag = False
        if random.random() < 1:
            x = TokenReplacement.mutate_feature_dif(x)
            crossover_flag = True
            # 交叉操作
            x, padding_length = TokenReplacement.create_individual(x)
            x, crossed_indices = TokenReplacement.crossover(x)

        y = self.ssm(x,z)
        if crossover_flag:
            # 逆交叉操作
            y = TokenReplacement.reverse_crossover(y, crossed_indices)
            y = TokenReplacement.remove_padding(y, padding_length)

        # x_b branch mutation
        x_b = x_b.transpose(1, 2)  # (B, ED, L)
        x_b = self.conv1d_b(x_b)[:, :, :L]  # depthwise convolution over time, with a short filter
        x_b = x_b.transpose(1, 2)  # (B, L, ED)
        x_b = F.silu(x_b)
        if random.random() <0.8:
            x_b = TokenReplacement.mutate_feature_dif(x_b)

        y_b = self.ssm_b(x_b)

        # z branch
        z = F.silu(z)
        output = y_b * z + y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def ssm_b(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)
        A = -torch.exp(self.A_log_b.float())  # (ED, N)
        D = self.D_b.float()

        deltaBC = self.x_proj_b(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj_b.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj_b.bias)
        y = self.selective_scan_b(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D, isdecoder=False):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N) 50,49,1024,16
        hs = pscan(deltaA, BX)
        hs = self.attention(hs)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y
    def selective_scan_b(self, x, delta, A, B, C, D, isdecoder=False):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)
        hs = pscan(deltaA, BX)
        hs = self.attention_b(hs)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x
        return y


    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y




# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
