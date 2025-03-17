import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model', '../../ubisoft-laforge-ZeroEGGS-main', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]
from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
import subprocess
import os
from datetime import datetime
from mfcc import MFCC
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_zeggs_bvh import pose2bvh, quat      # '../process'
import argparse

import numpy as np

import torch
import numpy as np
from torch.nn import functional as F
from scipy.signal import savgol_filter


import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import torch
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
import os
import sys
[sys.path.append(i) for i in ['.', '..', '../model', '../train']]
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
#from training_loop import TrainLoop
from model.mdm import MDM

def batch_rot2aa(Rs):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);

        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q



def wavlm_init(device=torch.device('cuda:2')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))     # load the pre-trained checkpoints
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, device=torch.device('cuda:2')):
    wav_input_16khz = wav_input_16khz.to(device)
    rep = model.extract_features(wav_input_16khz)[0]
    rep = F.interpolate(rep.transpose(1, 2), size=72, align_corners=True, mode='linear').transpose(1, 2)
    return rep


def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=433, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                glob_rot=True, cond_mode = 'cross_local_attention3_style1', clip_version = 'ViT-B/32', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=128, n_seed=12)        # trans_enc, trans_dec, gru, mytrans_enc
    diffusion = create_gaussian_diffusion()
    return model, diffusion



from tqdm import tqdm

def inference(args, wavlm_model, audio, sample_fn, model, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=0, n_seed=12, style=None, seed=123456):

    torch.manual_seed(seed)
    eval_sum_index=0

    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device='cuda')
    test_loader = DataLoader(dataset=val_dataset, batch_size=64,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)
    pre_pose=[]
    tar_pose=[]
    audio_pose=[]
    upper_index=[3,6,9,12,13,14,15,16,17,18,19,20,21]
    lower_index=[0,1,2,4,5,7,8,10,11]
    body_index=[13,14,15,0,16,17,1,18,19,2,20,21,3,4,5,6,7,8,9,10,11,12]
    for batch in tqdm(test_loader):
        pose_seq, style, wavlm, text = batch

        pose_seq_reshape = pose_seq[:,:,:330].reshape(-1,72,55,6)
        tar_pose.append(pose_seq_reshape.detach().cpu().numpy())#([64, 72, 55, 6])
        pose_expression_reshape = pose_seq[:,:,330:430].reshape(-1,72,100)
        pose_jaw = pose_seq_reshape[:,:,22:25,:].reshape(-1,72,3*6)
        pose_expression_reshape = torch.cat((pose_expression_reshape,pose_jaw),-1)#100+3*6
        pose_trans_reshape = pose_seq[:,:,430:433].reshape(-1,72,3)


        pose_hands = pose_seq_reshape[:,:,25:,:].reshape(-1,72,30*6)
        # pose_upper = pose_seq_reshape[:,:,upper_index,:].reshape(-1,72,14*6)
        pose_upper = pose_seq_reshape[:,:,upper_index,:].reshape(-1,72,13*6)
        pose_lower = pose_seq_reshape[:,:,lower_index,:].reshape(-1,72,9*6)
        pose_lower = torch.cat((pose_lower,pose_trans_reshape),axis=-1)#-1,64,9*6+3
        pose_expression = pose_expression_reshape


        pose_seq = torch.cat((pose_upper,pose_hands,pose_lower,pose_expression),axis=-1)# 14*6    30*6     9*6+3     100

        motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(mydevice)

        model_kwargs_ = {'y': {}}
        model_kwargs_['y']['mask'] = (torch.zeros([64, 1, 1, n_frames]) < 1).to(mydevice)
        model_kwargs_['y']['style'] = torch.as_tensor(style).float().to(mydevice)
        model_kwargs_['y']['mask_local'] = torch.ones(64, args.n_poses).bool().to(mydevice)
        model_kwargs_['y']['audio'] = wavlm.to(torch.float32).to(mydevice)
        model_kwargs_['y']['text'] = text
        #audio_pose.append(audio.clone().detach().cpu().numpy())
        model_kwargs_['y']['seed'] = motion[..., 0:12]

        shape_ = (64, model.njoints, model.nfeats, args.n_poses)

        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )#([64, 421, 1, 72])

        pre_pose.append(sample)#53*6+3
        eval_sum_index+=1
        # if eval_sum_index>2:
        #     break
        
        # smoothing motion transition
        # if len(pre_pose) > 0 and n_seed != 0:
        #     last_poses = pre_pose[-1][..., -n_seed:]        # # (1, model.njoints, 1, n_seed)
        #     pre_pose[-1] = pre_pose[-1][..., :-n_seed]  # delete last 4 frames
        #     if smoothing:
        #         # Extract predictions
        #         last_poses_root_pos = last_poses[:, 53*6:53*6+3]        # (1, 3, 1, 8)
        #         # last_poses_root_rot = last_poses[:, 3:7]
        #         # last_poses_root_vel = last_poses[:, 7:10]
        #         # last_poses_root_vrt = last_poses[:, 10:13]
        #         next_poses_root_pos = sample[:, 53*6:53*6+3]        # (1, 3, 1, 88)
        #         # next_poses_root_rot = sample[:, 3:7]
        #         # next_poses_root_vel = sample[:, 7:10]
        #         # next_poses_root_vrt = sample[:, 10:13]
        #         root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
        #         predict_pos = next_poses_root_pos[..., 0]
        #         delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
        #         sample[:, 53*6:53*6+3] = sample[:, 53*6:53*6+3] - delta_pos

        #     for j in range(len(last_poses)):
        #         n = len(last_poses)
        #         prev = last_poses[..., j]
        #         next = sample[..., j]
        #         sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        

        # if n_seed != 0:
        #     out_list[-1] = out_list[-1][..., :-n_seed]
        #     out_list = [i.detach().data.cpu().numpy() for i in out_list]
        #     out_dir_vec = np.vstack(out_list)
        #     sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
        #     sampled_seq = sampled_seq[:, n_seed:]
        # else:
        #     out_list = [i.detach().data.cpu().numpy() for i in out_list]
        #     out_dir_vec = np.vstack(out_list)
        #     sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
    pre_pose = [i.detach().data.cpu().numpy() for i in pre_pose]
    tar_pose = np.array(tar_pose).reshape(-1,72,55,6)
    pre_pose = np.array(pre_pose).reshape(-1,1,433,1,72)
    print("pre_pose shape:",np.array(pre_pose).shape,"    np.array(tar_pose) shape:",np.array(tar_pose).shape)
    pre_pose = np.array(pre_pose)
    pre_pose=pre_pose[:,:,:330,:,:]
    pre_pose_upper=pre_pose[:,:,:13*6,:,:].reshape(-1,72,13,6)
    pre_pose_hands=pre_pose[:,:,13*6:43*6,:,:].reshape(-1,72,30,6)
    pre_pose_lower=pre_pose[:,:,43*6:52*6,:,:].reshape(-1,72,9,6)
    pre_pose_body=np.concatenate((pre_pose_upper,pre_pose_lower),axis=-2)
    pre_pose_body = pre_pose_body[:,:,body_index,:]
    #pre_poses_rot6d = np.concatenate((pre_pose_body,out_poses_rot6d_hands),axis=-2)
    pre_pose = np.array(tar_pose).copy()
    pre_pose[:,:,:22,:]=pre_pose_body
    pre_pose[:,:,25:,:]=pre_pose_hands

    print("pre_pose shape:",np.array(pre_pose).shape)#(1, 1, 421, 1, 72)  14*6    30*6     9*6+3     100
    print("tar_pose shape:",np.array(tar_pose).shape)#(1, 1, 72, 55, 6)
    #print("tar_pose shape:",np.array(audio_pose).shape)#(150, 64, 38400)
    
    

    
        
    

    np.savez("ours_hologest_50steps_beatx_res.npz",tar_pose=np.array(tar_pose),pre_pose=np.array(pre_pose))
    # np.savez("ours_hologest_50steps_beatx_res.npz",tar_pose=np.array(tar_pose),pre_pose=np.array(pre_pose),audio=np.array(audio_pose))
    exit()
    data_mean_ = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/NIPS_speaker_video/PantoMatrix/BEAT2/Ours_BEAT2_lmdb/mean.npz")['mean'].squeeze()
    data_std_ = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/NIPS_speaker_video/PantoMatrix/BEAT2/Ours_BEAT2_lmdb/std.npz")['std'].squeeze()

    data_mean = np.array(data_mean_).squeeze()
    data_std = np.array(data_std_).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    
    
    #3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 2, 4, 5, 7, 8, 10, 11
    #14,15,16,0,17,18,1,19,20,2,21,22,3,4,5,6,7,8,9,10,11,12,13
    body_index=[14,15,16,0,17,18,1,19,20,2,21,22,3,4,5,6,7,8,9,10,11,12,13]
    print("data_mean shape:",data_mean.shape,"      std shape:",std.shape)
    data_mean_6d = data_mean[:330].reshape(55,6)
    std_6d = std[:330].reshape(55,6)
    mean_hands = data_mean_6d[25:,:].reshape(-1)
    std_hands = std_6d[25:,:].reshape(-1)
    mean_upper = data_mean_6d[upper_index,:].reshape(-1)
    std_upper = std_6d[upper_index,:].reshape(-1)
    mean_lower = data_mean_6d[lower_index,:].reshape(-1)
    std_lower = std_6d[lower_index,:].reshape(-1)
    mean_lower_trans = np.concatenate((mean_lower,data_mean[430:]),axis=0)
    std_lower_trans = np.concatenate((std_lower,std[430:]),axis=0)
    mean_exp = data_mean[330:430]
    std_exp = std[330:430]
    # pose_hands = pose_seq_reshape[:,:,25:,:].reshape(-1,64,30*6)
    # pose_upper = pose_seq_reshape[:,:,upper_index,:].reshape(-1,64,14*6)
    # pose_lower = pose_seq_reshape[:,:,lower_index,:].reshape(-1,64,9*6)
    # pose_lower = torch.cat((pose_lower,pose_trans_reshape),axis=-1)#-1,64,9*6+3
    # pose_expression = pose_expression_reshape

    data_mean = np.concatenate((mean_upper,mean_hands,mean_lower_trans,mean_exp),axis=0)
    std = np.concatenate((std_upper,std_hands,std_lower_trans,std_exp),axis=0)

    # pose_seq = torch.cat((pose_upper,pose_hands,pose_lower,pose_expression),axis=-1)# 14*6    30*6     9*6+3     100

    print("sampled_seq[0] shape:",sampled_seq[0].shape,"       data_mean shape:",data_mean.shape)
    out_poses = np.multiply(sampled_seq[0], std) + data_mean
    print(out_poses.shape)#(56, 421)


    data_rot6d_flat = out_poses.copy()
    n_poses = data_rot6d_flat.shape[0]
    out_poses = np.zeros((n_poses, data_rot6d_flat.shape[1]))
    for i in range(out_poses.shape[1]):
        out_poses[:, i] = savgol_filter(data_rot6d_flat[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal


    #(56, 421)
    out_poses_rot6d_upper = out_poses[:,:14*6].reshape(-1,14,6)
    out_poses_rot6d_hands = out_poses[:,14*6:44*6].reshape(-1,30,6)
    out_poses_rot6d_lower = out_poses[:,44*6:53*6].reshape(-1,9,6)
    out_poses_trans = out_poses[:,53*6:53*6+3].reshape(-1,3)
    out_poses_exp = out_poses[:,53*6+3:].reshape(-1,100)
    
    out_poses_rot6d_body = np.concatenate((out_poses_rot6d_upper,out_poses_rot6d_lower),axis=1)
    out_poses_rot6d_body = out_poses_rot6d_body[:,body_index,:]
    
    out_poses_rot6d = torch.tensor(np.concatenate((out_poses_rot6d_body,out_poses_rot6d_hands),axis=1).reshape(-1,6)).cuda()
    out_poses_rotmat = rot6d_to_rotmat(out_poses_rot6d)#-1,3,3
    out_poses_aa = batch_rot2aa(out_poses_rotmat)#-1,3
    out_poses_aa = out_poses_aa.detach().cpu().numpy().reshape(-1,53,3)
    eyes_zeros = np.zeros((out_poses_aa.shape[0],2,3))
    out_poses_aa = np.concatenate((out_poses_aa[:,:23,:],eyes_zeros,out_poses_aa[:,23:,:]),axis=1)

    np.savez("beat2_our_seed3.npz",poses=out_poses_aa,trans=out_poses_trans,betas=np.zeros(10),expression=out_poses_exp,gender='male',mocap_framerate=30,)


def main(args, save_dir, model_path, audio_path=None, mfcc_path=None, audiowavlm_path=None, max_len=0):


    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()

    sample_fn = diffusion.p_sample_loop     # predict x_start

    style = [1, 0, 0, 0, 0, 0]

    wavlm_model = wavlm_init(mydevice)
    inference(args, wavlm_model, None, sample_fn, model, n_frames=max_len, smoothing=True, SG_filter=True, minibatch=True, skip_timesteps=0, style=style, seed=-9923)      # style2onehot['Happy']


if __name__ == '__main__':
    '''
    cd /ceph/hdd/yangsc21/Python/DSG/
    '''

    # audio_path = '../../../My/Test_audio/Example1/ZeroEGGS_cut.wav'
    # mfcc_path = "../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/valid/mfcc/015_Happy_4_mirror_x_1_0.npz"       # 010_Sad_4_x_1_0.npz
    # audiowavlm_path = "./015_Happy_4_x_1_0.wav"

    # prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    # save_dir = 'sample_' + prefix
    save_dir = 'sample_dir'

    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='/apdcephfs/share_1290939/shaolihuang/ykcheng/SIGASIA_realtime_A2G/HoloGest/main/mydiffusion_zeggs/configs/HoloGest.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--model_path', type=str, default='/apdcephfs/share_1290939/shaolihuang/ykcheng/SIGASIA_realtime_A2G/HoloGest/main/mydiffusion_zeggs/beatX_50steps_hologest_all_speakers/model000250000.pt')
    parser.add_argument('--audiowavlm_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))

    batch_size = 1

    main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len)

