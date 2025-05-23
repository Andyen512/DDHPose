## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False,  bonechain=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.bonechain = bonechain


    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)       
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # hir_bias = self.hirmap_mlp(self.prior_map)
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_xxc(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False, bonechain=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_xc = nn.Linear(dim, dim , bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.bonechain = bonechain



    def forward(self, x, xc=None, vis=False):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if xc==None:
            pass
        else:
            qkv_xc = self.qkv_xc(xc).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k_xc = qkv_xc[0]

        # Now x shape (3, B, heads, N, C//heads)       
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        if q.shape[-2]==17:
            for chain in self.bonechain:
                for idx in range(1,len(chain)-1):
                    ppidx = chain[idx-1]
                    pidx = chain[idx]
                    cidx = chain[idx+1]
                    attn[:,:,pidx,cidx] = (attn[:,:,pidx,cidx] + attn[:,:,ppidx,pidx]) /2.0
                    attn[:,:,cidx,pidx] = (attn[:,:,cidx,pidx] + attn[:,:,ppidx,pidx]) /2.0

        else:
            if self.comb==True:
                xc_attn = (q.transpose(-2, -1) @ k_xc) * self.scale
            elif self.comb==False:
                xc_attn = (q @ k_xc.transpose(-2, -1)) * self.scale         
            attn += xc_attn

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False, bonechain=None):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis, bonechain=bonechain)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis


    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x
    
class Block_xxc(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention_xxc, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False, bonechain=None):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis, bonechain=bonechain)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis


    def forward(self, x, xc=None, vis=False):
        if xc==None:
            x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
            x = x + self.drop_path(self.mlp(self.norm2(x)))   
        else:         
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(xc), vis=vis))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        self.is_train=is_train
        
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3 + 1 , embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )

        self.group = nn.Parameter(torch.zeros(1, 6, embed_dim))
        self.lev0_list = [14]
        self.lev1_list = [8,11,15]
        self.lev2_list = [1,9,12,]
        self.lev3_list = [0,2,5,10,13,16]
        self.lev4_list = [3,6]
        self.lev5_list = [4,7]

        bonechain = [[14,8,9,10],[14,11,12,13],[14,15,1,0],[14,15,1,2,3,4],[14,15,1,5,6,7],[14,15,1,16]]

        self.bonedic = {0:None, 1:'0,2,5,16', 2:'3', 3:'4', 4:None, 5:'6', 6:'7', 7:None, 8:'9', 9:'10', 10:None, 11:'12', 12:'13', 13:None, 14:'8,11,15', 15:'1', 16:None}

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks_0 = nn.ModuleList([            
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, bonechain=bonechain)])
        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block_xxc(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, bonechain=bonechain)
            for i in range(1,depth)])

        self.TTEblocks_0 = nn.ModuleList([            
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=False, currentdim=1, depth=depth, bonechain=bonechain)])
        self.TTEblocks = nn.ModuleList([
            Block_xxc(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth, bonechain=bonechain)
            for i in range(1,depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head_pose = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def STE_forward(self, x_2d, x_3d, t):

        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = rearrange(x, 'b f n c  -> (b f) n c', )
            ### now x is [batch_size, receptive frames, joint_num, 2 channels]
            x = self.Spatial_patch_to_embedding(x)
            # x_hirmap = self.Spatial_patch_to_embedding_hirmap(dir_hir_map)

            # Hierarchical embedding.
            for lev in range(6):
                lev_list = eval('self.lev{:}_list'.format(lev))
                for idx in lev_list:
                    x[:,idx,:] += self.group[0][lev:lev+1]
            # x = rearrange(x, 'bnew c n  -> bnew n c', )
            x += self.Spatial_pos_embed

            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1,f,n,1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = self.Spatial_patch_to_embedding(x)

            # Hierarchical encoding.
            for lev in range(6):
                lev_list = eval('self.lev{:}_list'.format(lev))
                for idx in lev_list:
                    x[:,idx,:] += self.group[0][lev:lev+1]
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)

        blk = self.STEblocks_0[0]
        x = blk(x)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)        

        blk = self.TTEblocks_0[0]
        x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x,xc):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape

        for i in range(0, self.block_depth-1):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x,xc)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        return x


    def forward(self, x_2d, x_3d_dir, x_3d_bone, t):
        x_3d = torch.cat((x_3d_dir,x_3d_bone), dim=-1)
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape

        x_2d, t = x_2d.float(), t.float()

        x = self.STE_forward(x_2d, x_3d, t,)

        x = self.TTE_foward(x)

        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        xc_list = []
        xc = x.clone()
        for idx in range(17):
            pidx = idx
            if self.bonedic[idx]:
                cidx = [int(x) for x in self.bonedic[idx].split(',')]
                xc_cidx = xc[:,:,cidx]
                xc_cidx = torch.cat((xc_cidx , xc[:,:,pidx:pidx+1]),dim=2).mean(2)
            else:
                xc_cidx = xc[:,:,pidx:pidx+1].squeeze(2)
            xc_list.append(xc_cidx)
        xc = torch.stack(xc_list,dim=2)

        x = self.ST_foward(x,xc)

        x_pos = self.head_pose(x)
        if self.is_train:
            x_pos = x_pos.view(b, f, n, -1)
        else:
            x_pos = x_pos.view(b, h, f, n, -1)

        return x_pos


