import torch
from torch import nn
import numpy as np
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import torch
import torch.nn as nn
from functools import partial
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        #url='https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x

class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """ TNT Block
    """
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(inner_dim)
            # self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
            #                      out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        # self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
        #                      out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, inner_tokens, outer_tokens):
        if self.has_inner:
            #print(inner_tokens.shape)
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))  # B*N, k*k, c
            #print(inner_tokens.shape)
            # inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))  # B*N, k*k, c
            # print(inner_tokens.shape)

            B, N, C = outer_tokens.size()
            #print(inner_tokens.reshape(B, N - 1, -1).shape)
            outer_tokens[:, 1:] = outer_tokens[:, 1:] + self.proj_norm2(
                self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1))))  # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            # tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            # outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            # outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens

class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #print(x.shape)
        x = self.unfold(x)  # B, Ck2, N
        #print(x.shape)

        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size)  # B*N, C, 16, 16
        #print(x.shape)
        x = self.proj(x)  # B*N, C, 8, 8
        #print(x.shape)
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)  # B*N, 8*8, C
        #print(x.shape)
        return x

class TNT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        #return outer_tokens[:, 0]
        return outer_tokens[:, 1:, :]

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def tnt_s_patch16_224(pretrained=False, **kwargs):
    patch_size = 16
    inner_stride = 4
    outer_dim = 384
    inner_dim = 24
    outer_num_heads = 6
    inner_num_heads = 4
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def tnt_b_patch16_224(pretrained=False, **kwargs):
    patch_size = 16
    inner_stride = 4
    outer_dim = 640
    inner_dim = 40
    outer_num_heads = 10
    inner_num_heads = 4
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

# if __name__ == '__main__':
#     model = tnt_b_patch16_224(pretrained=False)
#     input = torch.randn((1,3,224,224))
#     output = model(input)
#     print(output.shape)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=196, encoder_dim=2048]
                attn_mask: [batch_size, len_q=196, len_k=196]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=52, embed_dim=512]
                attn_mask: [batch_size, len_q=52, len_k=52]
        encoder-decoder attention:
                Q: [batch_size, 52, 512] from decoder
                K, V: [batch_size, 196, 2048] from encoder
                attn_mask: [batch_size, len_q=52, len_k=196]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        #if attention_method == "ByPixel":
        self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=640, QKVdim=64, n_heads=n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=640, dropout=dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, decoder_layers, vocab_size, embed_dim, dropout, n_heads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, n_heads) for _ in range(decoder_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
 
    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(51)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        decode_lengths = len(encoded_captions[0]) - 1  # Exclude the last one

        # dec_outputs: [batch_size, max_len=52, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=52, len_k=52], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 52, 52], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(len(encoded_captions[0])))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        #if self.attention_method == "ByPixel":
        dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, len(encoded_captions[0]), 196))).to(device) == torch.tensor(np.ones((batch_size, len(encoded_captions[0]), 196))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        alphas = {"dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, alphas

# class EncoderLayer(nn.Module):
#     def __init__(self, dropout, attention_method, n_heads):
#         super(EncoderLayer, self).__init__()
#         """
#         In "Attention is all you need" paper, dk = dv = 64, h = 8, N=6
#         """
#         if attention_method == "ByPixel":
#             self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
#             self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)
#         elif attention_method == "ByChannel":
#             self.enc_self_attn = Multi_Head_Attention(Q_dim=196, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
#             self.pos_ffn = PoswiseFeedForwardNet(embed_dim=196, d_ff=512, dropout=dropout)

#     def forward(self, enc_inputs, enc_self_attn_mask):
#         """
#         :param enc_inputs: [batch_size, num_pixels=196, 2048]
#         :param enc_outputs: [batch_size, len_q=196, d_model=2048]
#         :return: attn: [batch_size, n_heads=8, 196, 196]
#         """
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
#         enc_outputs = self.pos_ffn(enc_outputs)
#         return enc_outputs, attn


# class Encoder(nn.Module):
#     def __init__(self, n_layers, dropout, attention_method, n_heads):
#         super(Encoder, self).__init__()
#         if attention_method == "ByPixel":
#             self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
#         # self.dropout = nn.Dropout(p=dropout)
#         self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
#         self.attention_method = attention_method

#     def get_position_embedding_table(self):
#         def cal_angle(position, hid_idx):
#             x = position % 14
#             y = position // 14
#             x_enc = x / np.power(10000, hid_idx / 1024)
#             y_enc = y / np.power(10000, hid_idx / 1024)
#             return np.sin(x_enc), np.sin(y_enc)
#         def get_posi_angle_vec(position):
#             return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

#         embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(196)])
#         return torch.FloatTensor(embedding_table).to(device)

#     def forward(self, encoder_out):
#         """
#         :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
#         """
#         batch_size = encoder_out.size(0)
#         positions = encoder_out.size(1)
#         if self.attention_method == "ByPixel":
#             encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device))
#         # encoder_out = self.dropout(encoder_out)
#         # enc_self_attn_mask: [batch_size, 196, 196]
#         enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
#                               == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
#         enc_self_attns = []
#         for layer in self.layers:
#             encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
#             enc_self_attns.append(enc_self_attn)
#         return encoder_out, enc_self_attns

# class Vit(nn.Module):
#     """
#     Vit_Encoder.
#     """

    # def __init__(self):
    #     super(Vit, self).__init__()

    #     self.vit = torchvision.models.vit_b_16(pretrained=True)

    #     #self.fine_tune()

    # def forward(self, images):
    #     """
    #     Forward propagation.

    #     :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
    #     :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
    #     """
    #     output_feature = create_feature_extractor(self.vit, return_nodes = { "encoder.ln":"getitem_5"} )
    #     out = output_feature(images)['getitem_5'] # 32,197,768
    #     out = out[:, 1: ,:]
        
    #     return out

    # def fine_tune(self, fine_tune=True):
    #     """
    #     Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

    #     :param fine_tune: Allow?
    #     """
    #     for p in self.resnet.parameters():
    #         p.requires_grad = False
    #     # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    #     for c in list(self.resnet.children())[5:]:
    #         for p in c.parameters():
    #             p.requires_grad = fine_tune

class TntTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_layers, dropout=0.1, n_heads=8):
        super(TntTransformer, self).__init__()
        self.encoder = tnt_b_patch16_224(pretrained=False)
        self.decoder = Decoder(decoder_layers = decoder_layers, vocab_size = vocab_size, embed_dim = embed_dim, dropout = dropout, n_heads = n_heads)
        self.embedding = self.decoder.tgt_emb

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption_batch(self, encoder_out, max_len = 51, itos=None, stoi=None):
        """
        Generate captions batch for the given image features (enc_inputs).
        """
        # Initialize lists to store attention weights
        batch_size = encoder_out.size(0)
        
        #dec_self_attns_list = []
        #dec_enc_attns_list = []
        #start_token = torch.tensor(stoi['<sos>']).unsqueeze(0).unsqueeze(0).to(device)
        
        start_tokens = torch.full((batch_size, 1), stoi['<sos>']).to(device)
        captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

        # print(captions)
        # print(captions.shape)
        
        input_tokens = start_tokens

        for i in range(max_len):
            predictions, attns = self.decoder(encoder_out, input_tokens)
            # print(predictions)
            # print(predictions.shape)
            
            # alpha = attns["dec_enc_attns"][-1]
            # alpha = alpha[:, 0, i, :].view(batch_size, 1, 14, 14).squeeze(1)
            # print(alpha)

            predictions = predictions[:, i, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # print(predictions)
            # print(predictions.shape)
            
            predictions = torch.argmax(predictions, dim=1)
            # print(predictions)
            # print(predictions.shape)
            
            captions[:, i] = predictions
            # print(captions)
            
            predicted_next_token = predictions.unsqueeze(1)
            # print(predicted_next_token)
            input_tokens = torch.cat([input_tokens, predicted_next_token], dim=1)
            # print(input_tokens)
        
        return captions

    def generate_caption(self, encoder_out, max_len = 51, itos=None, stoi=None):
        """
        Generate captions batch for the given image features (enc_inputs).
        """
        print(encoder_out.shape)
        batch_size = encoder_out.size(0)

        # Initialize lists to store attention weights        
        #dec_self_attns_list = []
        #dec_enc_attns_list = []
        #start_token = torch.tensor(stoi['<sos>']).unsqueeze(0).unsqueeze(0).to(device)
        start_tokens = torch.full((1, 1), stoi['<sos>']).to(device)

        #start_tokens = torch.tensor(stoi['<sos>']).view(1, -1).to(device)
        captions = []
        alphas = []
        #captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

        # print(captions)
        # print(captions.shape)
        
        input_tokens = start_tokens

        for i in range(max_len):
            predictions, attns = self.decoder(encoder_out, input_tokens)
            # print(predictions)
            #print(predictions.shape)

            alpha = attns["dec_enc_attns"][-1]
            alpha = alpha[:, 0, i, :].view(batch_size, 1, 14, 14).squeeze(1)
            #print(alpha.shape)
            alphas.append(alpha.cpu().detach().numpy())
            #print(alphas)

            predictions = predictions[:, i, :].view(batch_size, -1).squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # print(predictions)
            #print(predictions.shape)
            
            predictions = torch.argmax(predictions, dim=1)
            # print(predictions)
            # print(predictions.shape)
            
            captions.append(predictions.item())

            #print(captions)
            
            predicted_next_token = predictions.unsqueeze(1)
            # print(predicted_next_token)
            input_tokens = torch.cat([input_tokens, predicted_next_token], dim=1)
            # print(input_tokens)
            if itos[predicted_next_token.item()] == "<eos>":
                break
        #print(captions)
        captions = [itos[idx] for idx in captions]
        return captions, alphas
