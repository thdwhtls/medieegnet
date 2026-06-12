import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from modeling_medieeg_helper import *
from functools import partial
import numpy as np


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class MediEEGNet(nn.Module):
    def __init__(self, spectral_dim=16, time_dim = 16, depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, finetune=False, scratch=False, is_teacher=False, args=None, **kwargs):
        super().__init__()

        self.encoder = VisionTransformerEncoder(
                 spectral_dim=spectral_dim, time_dim = time_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, is_teacher=is_teacher, args=args)

        # alignment constraint
        self.teacher = VisionTransformerEncoder(
                spectral_dim=spectral_dim, time_dim = time_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.seq_len
        self.finetune = finetune
        self.scratch = scratch
        self.is_teacher = is_teacher
        self.spectral_dim= spectral_dim
        self.time_dim = time_dim
        self.embed_dim = spectral_dim * time_dim


        self.pretext_neck = VisionTransformerNeck(num_classes= 50, embed_dim= 256, depth=args.decoder_depth,
            num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, init_std=init_std, args=args)

        # encoder to decoder projection, borrowed from mae.
        if args.decoder_embed_dim != self.embed_dim:
            self.encoder_to_decoder = nn.Linear(self.embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None
        
        self.feature_predictor = FeaturePredictor(num_classes=256, embed_dim=256, depth=args.feature_predictor_depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, init_std=init_std, args=args)
        self.projection_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 16, 16))
        trunc_normal_(self.mask_token, std=self.init_std)

        self.linear = nn.Linear(self.embed_dim, args.num_classes)

        ## whether to use 'rescale' to init the weight, borrowed from beit.
        if not args.fix_init_weight:
            self.apply(self._init_weights)
        self._init_teacher()
        
        
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.encoder.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    '''
    Input shape:
        x: [bs, 62, 200]
        bool_masked_pos: [bs, 62]
    '''
    def forward(self, x, bool_masked_pos, return_all_tokens=None, samples=None):
        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        if self.is_teacher:
            x_unmasked, feature = self.teacher(x, bool_masked_pos=bool_masked_pos)
            x_unmasked = x_unmasked[:, 1:, :, :]
            return self.feature_predictor(x_masked=None, x_unmasked=x_unmasked, pos_embed_masked=None,
                                          pos_embed_unmasked=None, batches=False)

        x_unmasked, feature = self.encoder(x, bool_masked_pos=bool_masked_pos)

        if self.finetune or self.scratch:
            batch_size, seq_len, spectral_dim, time_dim = x_unmasked.shape
            out = self.linear((x_unmasked.reshape(batch_size, seq_len, spectral_dim * time_dim))[:, 0, :])
            x_unmasked = x_unmasked[:, 1:, :, :]

            batch_features = self.feature_predictor(x_masked=None, x_unmasked=x_unmasked, pos_embed_masked=None,
                                                    pos_embed_unmasked=None, batches=False)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)


            if samples is None:
                return out, batch_features


            samples_features, _ = self.encoder(samples, bool_masked_pos=torch.zeros(samples.size(0), samples.size(1)).bool().to(samples.device))
            samples_features = samples_features[:, 1:, :]
            samples_features = self.feature_predictor(x_masked=None, x_unmasked=samples_features, pos_embed_masked=None,
                                                      pos_embed_unmasked=None, batches=False)

            logit_scale = self.logit_scale.exp()
            samples_features = samples_features / samples_features.norm(dim=-1, keepdim=True)
            logits_contra = logit_scale * batch_features @ samples_features.t()
            return out, logits_contra

        # encoder to decoder projection
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)

        '''
        Alignment constraint
        '''
        with torch.no_grad():
            latent_target, _ = self.teacher(x, bool_masked_pos=(~bool_masked_pos))

            latent_target = latent_target[:, 1:, :] # remove class token


            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

            if samples is not None:

                samples_features, _ = self.teacher(samples, bool_masked_pos=torch.zeros(samples.size(0), samples.size(1)).bool().to(samples.device))
                samples_features = samples_features[:,1:,:]
                samples_features = self.feature_predictor(x_masked = None, x_unmasked = samples_features, pos_embed_masked = None,pos_embed_unmasked = None, batches = False)  # .mean(dim=1)

            self.momentum_update(self.args.base_momentum)

        '''
        Latent contextual regressor and decoder
        '''
        b, num_visible_plus1, spectral_dim, time_dim = x_unmasked.shape


        # remove class token
        x_unmasked = x_unmasked[:, 1:, :, :]
        num_masked_patches = self.num_patches - (num_visible_plus1-1)

        # generate position embeddings.
        pos_embed = self.encoder.pos_embed.expand(batch_size, self.num_patches+1, spectral_dim, time_dim).cuda(x_unmasked.device)

        # pos embed for masked patches
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, spectral_dim, time_dim)

        # pos embed for unmasked patches
        pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, spectral_dim, time_dim)

        # masked embedding '''
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, spectral_dim, time_dim)



        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos)
        batch_features = self.feature_predictor(x_masked, x_unmasked, pos_embed_masked,pos_embed_unmasked, batches = True)  # .mean(dim=1)

        if samples is None:
            return logits, latent_pred, latent_target
        logit_scale = self.logit_scale.exp()
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        samples_features = samples_features / samples_features.norm(dim=-1, keepdim=True)

        logits_contra = logit_scale * batch_features @ samples_features.t()
        return logits, latent_pred, latent_target, logits_contra
