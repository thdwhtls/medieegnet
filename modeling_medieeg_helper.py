import math
import torch
from multiprocessing import pool
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path
from einops import rearrange, repeat
from torch.nn.utils import weight_norm
import numpy as np


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class SpectralEmbedding(nn.Module):
    def __init__(self, bands_num = 5, spectral_dim = 16):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.bands_num = bands_num
        self.linear = nn.Linear(bands_num, spectral_dim)

    def forward(self, x):
        batch_size,channel_num, bands_num, time_step = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, bands_num)
        x = self.linear(x)
        x = x.reshape(batch_size, channel_num, time_step, self.spectral_dim)
        x = x.permute(0, 1, 3, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_save = None

    def forward(self, x, bool_masked_pos=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)


        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn_save = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

'''
Modified from Attention()
'''
class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, k=None, v=None):
        B, N, C= x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding),
                              stride=(1, stride))
        self.delect_pad = delect_padding(padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanh =nn.Tanh()
        self.gelu = nn.GELU()
        nn.init.constant_(self.conv.bias, 0.0)
        self.conv.weight.data.normal_(0,0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.delect_pad(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class delect_padding(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn1 = unit_tcn(16,16, kernel_size= 2, stride = 1)
        self.tcn2 = unit_tcn(16, 16, kernel_size=5, stride=1)
        self.tcn3 = unit_tcn(16, 16, kernel_size=7, stride=1)
    def forward(self, x):
        return self.tcn1(x) + self.tcn2(x) + self.tcn3(x)




class TCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation = 1, dropout = 0.):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels= n_inputs,
            out_channels = n_outputs,
            kernel_size = (1, kernel_size),
            stride = (1, stride),
            padding = (0, (kernel_size - 1) // 2 * dilation)
        )
        self.bn = nn.BatchNorm2d(n_outputs)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, bool_masked_pos=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), bool_masked_pos))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), bool_masked_pos))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones((dim)), requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones((dim)), requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q), bool_masked_pos, k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x


'''
Encoder that extracts representations
'''
class VisionTransformerEncoder(nn.Module):
    def __init__(self, spectral_dim=16, time_dim = 16, depth=6, seq_len=62,
                 num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, is_teacher=False, args=None, **kwargs):
        super().__init__()
        self.num_features = self.spectral_dim = spectral_dim
        self.seq_len = seq_len
        self.spectral_dim = spectral_dim
        self.time_dim = time_dim
        self.embed_dim = spectral_dim * time_dim
        self.is_teacher = is_teacher

        self.temporalBlock = nn.Sequential(
            TemporalBlock(),
        )
        # generate class token and pos embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spectral_dim, time_dim))
        self.pos_embed = self.build_position_embedding(spectral_dim, use_cls_token=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spectral_embedding_layer = SpectralEmbedding(spectral_dim = 16)
        self.tcn = nn.Sequential(
            TCN(n_inputs = 1, n_outputs = 16, kernel_size = 3, stride = 1, dilation = 1),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None,
                attn_head_dim=attn_head_dim
            )
            for i in range(depth)])
        self.norm = norm_layer(self.embed_dim)

        self.init_std = init_std

        # init the model
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        # rescale init function from beit
        # if it is not activated, it will be overwritten
        self.fix_init_weight()

    def build_position_embedding(self, spectral_dim=16, use_cls_token=False):
        pos_emb = torch.zeros((1, self.seq_len, spectral_dim, self.time_dim))

        omega = np.arange(spectral_dim // 2, dtype=np.float64)
        omega /= spectral_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = np.arange(self.seq_len)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

        emb = torch.from_numpy(emb).float()


        pos_emb[0, :, :, :] = emb.unsqueeze(-1).expand(-1, -1, self.time_dim)

        if not use_cls_token:
            pos_embed = nn.Parameter(pos_emb)
        else:
            pe_token = torch.zeros([1, 1, spectral_dim, self.time_dim], dtype=torch.float32)
            pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = True
        return pos_embed


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x = x.float()
        x = self.spectral_embedding_layer(x).to(device)       # batch_size * channel_num * dim * time_step

        x = x.permute(0, 2, 1, 3)
        x = self.temporalBlock(x)  # (batch_size * channel_num) * 16 * dim * 1
        x = x.permute(0, 2, 1, 3)
        batch_size, seq_len, dim, time_step = x.size()

        x = x.reshape(-1, 1, dim, time_step)     #(batch_size * channel_num) * 1 * dim * time_step


        x = self.tcn(x)       # (batch_size * channel_num) * 16 * dim * 1
        x = x.reshape(batch_size, seq_len, 16, dim)      # batch_size * channel_num * 16 * dim

        x = x.permute(0,1,3,2)        #batch_size * channel_num * spectral_dim * time__dim
        batch_size, channel_num, spectral_dim,  time_dim = x.shape


        # unmasked embeddings
        x_unmasked = x[~bool_masked_pos].reshape(batch_size, -1, spectral_dim, time_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1, -1)

        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(batch_size, self.seq_len+1, self.spectral_dim, self.time_dim)
            pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, self.spectral_dim, self.time_dim)
            pos_embed_unmasked = torch.cat((pos_embed[:,:1], pos_embed_unmasked),dim=1)
            x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.pos_drop(x_unmasked)

        batch_size, seq_len, spectral_dim, time_dim = x_unmasked.shape
        x_unmasked = x_unmasked.reshape(batch_size, seq_len, -1)
        for i, blk in enumerate(self.blocks):
            x_unmasked = blk(x_unmasked, bool_masked_pos)
            if i == 2 and not self.is_teacher:
                feature = x_unmasked.reshape(batch_size, seq_len, spectral_dim, time_dim)
            elif i == 5 and self.is_teacher:
                feature = x_unmasked.reshape(batch_size, seq_len, spectral_dim, time_dim)
        x_unmasked = self.norm(x_unmasked)
        x_unmasked = x_unmasked.reshape(batch_size, seq_len, spectral_dim, time_dim)
        return x_unmasked, feature

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        return x

'''
Latent context regressor + decoder that solves the pretext task.
'''
class VisionTransformerNeck(nn.Module):
    def __init__(self, num_classes, embed_dim=256, depth=3,
                 num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, init_std=0.02, args=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        self.args = args
        self.num_classes = num_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # context regressor
        self.regressor_blocks = nn.ModuleList([
            RegressorBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(args.regressor_depth)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, args.decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(args.decoder_depth)])

        self.norm = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.init_std = init_std

        # init the model
        trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.regressor_blocks):
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight.data, layer_id + 1)

        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
        
    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos):                
        # latent contextual regressor
        batch_size, seq_len, spectral_dim, time_dim = x_unmasked.shape
        embed_dim = spectral_dim * time_dim
        x_unmasked = x_unmasked.reshape(batch_size, seq_len, -1)
        x_masked = x_masked.reshape(batch_size, -1, embed_dim)
        pos_embed_unmasked = pos_embed_unmasked.reshape(batch_size, -1, embed_dim)
        pos_embed_masked = pos_embed_masked.reshape(batch_size, -1, embed_dim)
        for blk in self.regressor_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1), bool_masked_pos)
        x_masked = self.norm(x_masked)
        latent_pred = x_masked.reshape(batch_size, -1, spectral_dim, time_dim)
        
        x_masked = x_masked + pos_embed_masked  # add pos embed, like encoder
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm2(x_masked)
        logits = self.head(x_masked).reshape(batch_size, -1, self.num_classes)
        return logits, latent_pred


'''
Feature predictor.
'''
class FeaturePredictor(nn.Module):
    def __init__(self, num_classes=256, embed_dim=256, depth=3,
                 num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, init_std=0.02, args=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        self.args = args

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, args.feature_predictor_depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.relu = nn.ReLU()

        self.init_std = init_std

        # init the model
        trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, batches = False):
        if batches == True:
            x_unmasked = x_unmasked + pos_embed_unmasked
            x_masked = x_masked + pos_embed_masked
            x = torch.cat([x_masked, x_unmasked], dim=1)
        else:
            x = x_unmasked

        batch_size, seq_len, spectral_dim, time_dim = x.shape
        x = x.reshape(batch_size, -1, spectral_dim * time_dim)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.norm(x)

        logits = self.relu(self.head(x.mean(dim = 1)))
        return logits

