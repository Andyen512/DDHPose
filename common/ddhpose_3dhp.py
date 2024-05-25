import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from common.arguments import parse_args
from common.mixste_ddhpose_3dhp import *

__all__ = ["DDHPose"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise_dir', 'pred_noise_bone', 'pred_x_start'])
args = parse_args()
boneindextemp = args.boneindex_3dhp.split(',')
boneindex = []
for i in range(0,len(boneindextemp),2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i+1])])
    

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[1]] - seq[:,index[0]])
    bonedirect = torch.stack(bone,1)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect/bonesum
    bonedirect = bonedirect.view(bs,ss,-1,3)
    return bonedirect

def getbonedirect_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:,:,:,index[1]] - seq[:,:,:,index[0]])
    bonedirect = torch.stack(bone,3)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(-1), 0.5).unsqueeze(-1)
    bonedirect = bonedirect/bonesum
    return bonedirect

def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[1]] - seq[:,index[0]])
    bone = torch.stack(bone,1)
    bone = torch.pow(torch.pow(bone,2).sum(2),0.5)
    bone = bone.view(bs,ss, bone.size(1),1)
    return bone

def getbonelength_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:,:,:,index[1]] - seq[:,:,:,index[0]])
    bone = torch.stack(bone,3)
    bone = torch.pow(torch.pow(bone,2).sum(-1),0.5).unsqueeze(-1)
    return bone

class DDHPose(nn.Module):
    """
    Implement D3DP
    """

    def __init__(self, args, joints_left, joints_right, is_train=True, num_proposals=1, sampling_timesteps=1):
        super().__init__()

        self.frames = args.number_of_frames
        self.num_proposals = num_proposals
        self.flip = args.test_time_augmentation
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.is_train = is_train

        # build diffusion
        timesteps = args.timestep
        #timesteps_eval = args.timestep_eval
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        #self.num_timesteps_eval = int(timesteps_eval)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args.scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        #self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        drop_path_rate=0
        if is_train:
            drop_path_rate=0.1

        self.dir_bone_estimator = MixSTE2(num_frame=self.frames, num_joints=17, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=drop_path_rate, is_train=is_train)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions_dir_bone(self, x_dir, x_bone, inputs_2d, input_2d_flip, t):
        x_t_dir = torch.clamp(x_dir, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t_dir = x_t_dir / self.scale
        x_t_bone = torch.clamp(x_bone, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t_bone = x_t_bone / self.scale

        pred_pose = self.dir_bone_estimator(inputs_2d, x_t_dir, x_t_bone, t)

        # input 2d flip
        x_t_dir_flip = x_t_dir.clone()
        x_t_dir_flip[:, :, :, :, 0] *= -1
        x_t_dir_flip[:, :, :, self.joints_left + self.joints_right] = x_t_dir_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]
        x_t_bone_flip = x_t_bone.clone()
        x_t_bone_flip[:, :, :, self.joints_left + self.joints_right] = x_t_bone_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]

        pred_pose_flip = self.dir_bone_estimator(input_2d_flip, x_t_dir_flip, x_t_bone_flip, t)
        
        pred_pose_flip[:, :, :, :, 0] *= -1
        pred_pose_flip[:, :, :, self.joints_left + self.joints_right] = pred_pose_flip[:, :, :,
                                                                      self.joints_right + self.joints_left]

        pred_pos = (pred_pose + pred_pose_flip) / 2

        x_start_dir = getbonedirect_test(pred_pos,boneindex)
        x_start_dir = x_start_dir * self.scale
        x_start_dir = torch.clamp(x_start_dir, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise_dir = self.predict_noise_from_start(x_dir[:,:,:,1:,:], t, x_start_dir)

        x_start_bone = getbonelength_test(pred_pos,boneindex)
        x_start_bone = x_start_bone * self.scale
        x_start_bone = torch.clamp(x_start_bone, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise_bone = self.predict_noise_from_start(x_bone[:,:,:,1:,:], t, x_start_bone)

        x_start_pos = pred_pos
        x_start_pos = x_start_pos * self.scale
        x_start_pos = torch.clamp(x_start_pos, min=-1.1 * self.scale, max=1.1*self.scale)

        return ModelPrediction(pred_noise_dir, pred_noise_bone, x_start_pos)

    def ddim_sample_bone_dir(self, inputs_2d, inputs_3d, clip_denoised=True, do_postprocess=True, input_2d_flip=None):
        batch = inputs_2d.shape[0]
        jt_num = inputs_2d.shape[-2]
        dir_shape = (batch, self.num_proposals, self.frames, jt_num, 3)
        bone_shape = (batch, self.num_proposals, self.frames, jt_num, 1)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img_dir = torch.randn(dir_shape, device='cuda')
        img_bone = torch.randn(bone_shape, device='cuda')

        x_start_dir = None
        x_start_bone = None

        preds_all_pos = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()
            # self_cond = x_start if self.self_condition else None

            #print("%d/%d" % (time, total_timesteps))
            preds_pos = self.model_predictions_dir_bone(img_dir, img_bone, inputs_2d, input_2d_flip, time_cond)
            pred_noise_dir, pred_noise_bone, x_start_pos = preds_pos.pred_noise_dir, preds_pos.pred_noise_bone, preds_pos.pred_x_start

            x_start_dir = getbonedirect_test(x_start_pos,boneindex)
            x_start_bone = getbonelength_test(x_start_pos,boneindex)
            
            preds_all_pos.append(x_start_pos)

            if time_next < 0:
                img_dir = x_start_dir
                img_bone = x_start_bone
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise_dir = torch.randn_like(x_start_dir)
            noise_bone = torch.randn_like(x_start_bone)

            img_dir_t = x_start_dir * alpha_next.sqrt() + \
                    c * pred_noise_dir + \
                    sigma * noise_dir
            img_bone_t = x_start_bone * alpha_next.sqrt() + \
                    c * pred_noise_bone + \
                    sigma * noise_bone


            img_dir[:,:,:,:14] = img_dir_t[:,:,:,:14]
            img_dir[:,:,:,15:] = img_dir_t[:,:,:,14:]
            img_bone[:,:,:,:14] = img_bone_t[:,:,:,:14]
            img_bone[:,:,:,15:] = img_bone_t[:,:,:,14:]

        return torch.stack(preds_all_pos, dim=1)*1000

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, input_2d, input_3d, input_2d_flip=None):

        # Prepare Proposals.
        if not self.is_train:
            pred_pose = self.ddim_sample_bone_dir(input_2d, input_3d, input_2d_flip=input_2d_flip)
            return pred_pose

        if self.is_train:
            input_3d = input_3d / 1000

            x_dir, dir_noises, x_bone_length, bone_length_noises, t = self.prepare_targets(input_3d)
            x_dir = x_dir.float()
            x_bone_length = x_bone_length.float()

            t = t.squeeze(-1)

            pred_pose = self.dir_bone_estimator(input_2d, x_dir, x_bone_length, t)

            return pred_pose*1000


    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(self.frames, 17, 3, device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min= -1.1 * self.scale, max= 1.1*self.scale)
        x = x / self.scale


        return x, noise, t

    def prepare_diffusion_bone_dir(self, dir, bone):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise_dir = torch.randn(self.frames, dir.shape[1], dir.shape[2], device='cuda')
        noise_bone = torch.randn(self.frames, bone.shape[1], bone.shape[2], device='cuda')

        x_start_dir = dir
        x_start_bone = bone

        x_start_dir = x_start_dir * self.scale
        x_start_bone = x_start_bone * self.scale

        # noise sample
        x_dir = self.q_sample(x_start=x_start_dir, t=t, noise=noise_dir)
        x_bone = self.q_sample(x_start=x_start_bone, t=t, noise=noise_bone)

        x_dir = torch.clamp(x_dir, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_dir = x_dir / self.scale
        x_bone = torch.clamp(x_bone, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_bone = x_bone / self.scale


        return x_dir, noise_dir, x_bone, noise_bone, t

    def prepare_targets(self, targets):
        diffused_dir = []
        noises_dir = []
        diffused_bone_length = []
        noises_bone_length = []
        ts = []
        
        targets_dir = torch.zeros(targets.shape[0],targets.shape[1],targets.shape[2],3).cuda()
        targets_bone_length = torch.zeros(targets.shape[0],targets.shape[1],targets.shape[2],1).cuda()

        dir = getbonedirect(targets,boneindex)
        bone_length = getbonelength(targets,boneindex)

        targets_dir[:,:,:14] = dir[:,:,:14]
        targets_dir[:,:,15:] = dir[:,:,14:]
        targets_bone_length[:,:,:14] = bone_length[:,:,:14]
        targets_bone_length[:,:,15:] = bone_length[:,:,14:]

        for i in range(0,targets.shape[0]):
            targets_per_sample_dir = targets_dir[i]
            targets_per_sample_bone_length = targets_bone_length[i]

            d_dir, d_noise_dir, d_bone_length, d_noise_bone_length, d_t = self.prepare_diffusion_bone_dir(targets_per_sample_dir, targets_per_sample_bone_length)

            diffused_dir.append(d_dir)
            noises_dir.append(d_noise_dir)

            diffused_bone_length.append(d_bone_length)
            noises_bone_length.append(d_noise_bone_length)
            ts.append(d_t)

        return torch.stack(diffused_dir), torch.stack(noises_dir),  \
            torch.stack(diffused_bone_length), torch.stack(noises_bone_length), torch.stack(ts)


