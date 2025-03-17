import numpy as np
import os


import torch
import numpy as np
from torch.nn import functional as F




from math import pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

import sys
import json
import joblib

import json
import csv


import numpy as np
from remedian import remedian

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
[sys.path.append(i) for i in ['.', '..', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]

from anim import bvh, quat, txform
from utils_zeggs import write_bvh
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


from scipy.spatial.transform import Rotation
from beat_data_proc.MyBVH import load_bvh_data

from torch.nn import functional as F

def batch_rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
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



def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
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

def euler2mat(angles, euler_orders):
    assert angles.ndim == 3 and angles.shape[2] == 3, f"wrong shape: {angles.shape}"
    assert angles.shape[1] == len(euler_orders)

    nJoints = len(euler_orders)
    nFrames = len(angles)
    rot_mats = np.zeros((nFrames, nJoints, 3, 3), dtype=np.float32)

    for j in range(nJoints):
        # {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations
        R = Rotation.from_euler(euler_orders[j].upper(), angles[:, j, :], degrees=True)  # upper for intrinsic rotation
        rot_mats[:, j, :, :] = R.as_matrix()
    return rot_mats


def pose2json_directly(poses, outpath, length, smoothing=False, smooth_foot=False, upper=False):
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15,
    12, 17, 18, 19, 20, 17, 22, 23, 24, 25, 12, 27, 28, 29, 30, 27, 32,
    33, 34, 4, 36, 37, 38, 39, 40, 41, 42, 39, 44, 45, 46, 47, 44, 49,
    50, 51, 52, 39, 54, 55, 56, 57, 54, 59, 60, 61, 0, 63, 64, 65, 66,
    67, 0, 69, 70, 71, 72, 73], dtype=np.int32)
    order = 'zyx'
    dt = 0.05
    njoints = 75

    # smoothing
    if smoothing:
        n_poses = poses.shape[0]
        out_poses = np.zeros((n_poses, poses.shape[1]))
        for i in range(out_poses.shape[1]):
            # if (13 + (njoints - 14) * 9) <= i < (13 + njoints * 9): out_poses[:, i] = savgol_filter(poses[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
            # else:
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses
    # Extract predictions
    P_root_pos = out_poses[:, 0:3]
    P_root_rot = out_poses[:, 3:7]
    P_lpos = out_poses[:, 13 + njoints * 0: 13 + njoints * 3].reshape([length, njoints, 3])
    P_ltxy = out_poses[:, 13 + njoints * 3: 13 + njoints * 9].reshape([length, njoints, 2, 3])

    P_ltxy = torch.as_tensor(P_ltxy, dtype=torch.float32)
    P_lrot = quat.from_xform(txform.xform_orthogonalize_from_xy(P_ltxy).cpu().numpy())        #


    P_lpos = P_lpos.copy()
    P_lrot = P_lrot.copy()
    P_lpos[:, 0] = quat.mul_vec(P_root_rot, P_lpos[:, 0]) + P_root_pos
    P_lrot[:, 0] = quat.mul(P_root_rot, P_lrot[:, 0])
    P_lrot[:,[65,66,67,68,71,72,73,74],:]*=0.0
    transl = np.array(P_lpos[:, 0])*0.01
    quat_num = np.array(P_lrot)
    beats_aa=quaternion_to_angle_axis(torch.tensor(quat_num).reshape(-1,4)).numpy().reshape(-1,75,3)

    corr_index =  np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54])
    smplx_index = np.array([0,69,63,1,70,64,2,71,65,3,72,66,4,36,9,5,37,10,38,11,39,12,55,56,57,40,41,42,50,51,52,45,46,47,59,60,61,28,29,30,13,14,15,23,24,25,18,19,20,32,33,34])
    smplx_aa = np.zeros((len(beats_aa),55,3))
    smplx_aa[:,corr_index,:]=beats_aa[:,smplx_index,:]
    quat2json(smplx_aa,transl)

# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import torch.nn as nn



import enum
class QuaternionCoeffOrder(enum.Enum):
    XYZW = 'xyzw'
    WXYZ = 'wxyz'

class PCG():
    def __init__(self, face_smoother=None):
        self.face_smoother = face_smoother
        self.bone_list = []
        self.pose_list = []
        self.rota_list = []
        self.buffer = []
        self.bone_name_from_index = {
            0: 'pelvis',#'Pelvis', XYZ -> (-X)(-Y)Z, (5) XYZ
            1: 'left_hip',#'L_Hip', XYZ -> (-X)(-Y)Z, 后外高 -> 前里高 (3) XYZ
            2: 'right_hip',#'R_Hip', (4) XYZ -> (-X)(-Y)Z, 后里低 -> 前外低 (4) XYZ
            3: 'spine1',#'Spine1', (-X)Y(-Z) -> (0) XYZ
            4: 'left_knee',#'L_Knee', 同左UpperLeg
            5: 'right_knee',#'R_Knee',同右UpperLeg
            6: 'spine2',
            7: 'left_ankle',
            8: 'right_ankle',#'R_Ankle',同右UpperLeg
            9: 'spine3',#'Spine3', (-X)Y(-Z) 同脊椎
            10: 'left_foot',#'L_Foot',同左UpperLeg
            11: 'right_foot',#'R_Foot',同右UpperLeg
            12: 'neck',#'Neck', (-X)Y(-Z) 同脊椎
            13: 'left_collar',#'L_Collar', XYZ -> ZXY (VRM), 前拧, 后, 高 -> 高, 前拧, 后 (1) YZX
            14: 'right_collar',#'R_Collar', XYZ -> (-Z)(-X)Y , 前拧, 前, 低 -> 高, 后拧, 前 (2) YZX
            15: 'head',#'Head', (-X)Y(-Z) 同脊椎
            16: 'left_shoulder',#'L_Shoulder', 同左肩膀
            17: 'right_shoulder',#'R_Shoulder', 同右肩膀
            18: 'left_elbow',#'L_Elbow', 同左肩膀
            19: 'right_elbow',#'R_Elbow', 同右肩膀

            # writst, 2 part
            20: 'left_wrist',#'L_Wrist', 同左肩膀
            21: 'right_wrist',#'R_Wrist', 同右肩膀

            # left hand 15
            22: 'left_wrist',#'L_Wrist', 同左肩膀
            23: 'left_index1',
            24: 'left_index2',
            25: 'left_index3',

            26: 'left_middle1',
            27: 'left_middle2',
            28: 'left_middle3',

            29: 'left_pinky1',
            30: 'left_pinky2',
            31: 'left_pinky3',

            32: 'left_ring1',
            33: 'left_ring2',
            34: 'left_ring3',

            35: 'left_thumb1',
            36: 'left_thumb2',
            37: 'left_thumb3',


            # left hand 15
            38: 'right_wrist',#'R_Wrist', 同右肩膀
            39: 'right_index1',
            40: 'right_index2',
            41: 'right_index3',

            42: 'right_middle1',
            43: 'right_middle2',
            44: 'right_middle3',

            45: 'right_pinky1',
            46: 'right_pinky2',
            47: 'right_pinky3',

            48: 'right_ring1',
            49: 'right_ring2',
            50: 'right_ring3',

            51: 'right_thumb1',
            52: 'right_thumb2',
            53: 'right_thumb3',

            54: 'left_eye_smplhf',
            55: 'right_eye_smplhf'
        }
        self.init_translation = []
        self.buffer_json = []
        # self.buffer_json = {}
        # self.buffer_json['trans'] = []
        # for i in self.bone_name_from_index.keys():
        #     self.buffer_json[self.bone_name_from_index[i]] = []

        #body_used_id = np.array([0,1,4,7,10,2,5,8,11,3,6,9,13,16,18,20,14,17,19,21,12,15])
        self.body_used_id = np.array([0,1,4,7,10,2,5,8,11,3,6,9,13,16,18,22,14,17,19,38,12,15])
        self.hand_used_id = np.array([35,36,37,23,24,25,26,27,28,29,30,31,32,33,34, 
                                51,52,53,42,43,44,39,40,41,45,46,47,48,49,50])

        self.used_id = np.concatenate((self.body_used_id, self.hand_used_id))

        self.protocol_0 = [3, 6, 9, 12, 15]#['J_Bip_C_Spine', 'J_Bip_C_Chest', 'J_Bip_C_UpperChest', 'J_Bip_C_Neck', 'J_Bip_C_Head'] XYZ

        self.protocol_1 = [13, 16, 18]#['J_Bip_L_Shoulder', 'J_Bip_L_UpperArm', 'J_Bip_L_LowerArm', 'J_Bip_L_Hand'] YZX
        self.protocol_2 = [14, 17, 19]#['J_Bip_R_Shoulder', 'J_Bip_R_UpperArm', 'J_Bip_R_LowerArm', 'J_Bip_R_Hand'] YZX

        self.protocol_3 = [1, 4,   7,  10]#['J_Bip_L_UpperLeg', 'J_Bip_L_LowerLeg', 'J_Bip_L_Foot', 'J_Bip_L_ToeBase'] XYZ
        self.protocol_4 = [2, 5,   8,  11]#['J_Bip_R_UpperLeg', 'J_Bip_R_LowerLeg', 'J_Bip_R_Foot', 'J_Bip_R_ToeBase'] XYZ

        self.protocol_5 = [0]
        self.left_hand = np.arange(22, 38)
        self.right_hand = np.arange(38, 54)
        self.last_face = None

    def export_txt(self, out_path="C:/Users/Administrator/Desktop/lab.txt"):
        # export to txt
        with open(out_path, "w") as f:
            f.write("\n".join(self.buffer))
        return

    def export_json(self, out_path="C:/Users/Administrator/Desktop/lab.json"):
        # export to txt
        with open(out_path, "w") as f:
            json.dump(self.buffer_json, f)
        return
    
    def export_gilly_new(self, rot, bone_name, idx):
        if idx in self.protocol_5:

            #rot[0], rot[1], rot[2], rot[3] = -rot[1], rot[0], -rot[3], rot[2] # rotate x 180 for pare
            rot[1], rot[2], rot[3] = rot[1], -rot[2], -rot[3]
        else:
            rot[1], rot[2], rot[3] = rot[1], -rot[2], -rot[3]

        # transform coordinate system
        rot[1] = -rot[1]
        rot[2] = -rot[2]

        return rot.tolist()

    def export_ue(self, rot, bone_name, idx):

        if idx in self.protocol_0:
            rot[1], rot[2], rot[3] = rot[3], rot[1], rot[2]

        elif idx in self.protocol_1:
            if idx == 16: # shoulder offset for A-Pose model
                rot[1], rot[2], rot[3] = -rot[2], -rot[3], rot[1]
                rot[0], rot[1], rot[2], rot[3] = 0.94*rot[0]+0.342*rot[2], 0.94*rot[1]-0.342*rot[3], 0.94*rot[2]-0.342*rot[0], 0.94*rot[3]+0.342*rot[1]
            else:
                rot[1], rot[2], rot[3] = -rot[2], -rot[3], rot[1]

        elif idx in self.protocol_2:
            if idx == 17: # shoulder offset for A-Pose model
                rot[1], rot[2], rot[3] = rot[2], rot[3], rot[1]
                rot[0], rot[1], rot[2], rot[3] = 0.94*rot[0]+0.342*rot[2], 0.94*rot[1]-0.342*rot[3], 0.94*rot[2]-0.342*rot[0], 0.94*rot[3]+0.342*rot[1]
            else:
                rot[1], rot[2], rot[3] = rot[2], rot[3], rot[1]

        elif idx in self.protocol_3:
            rot[1], rot[2], rot[3] = -rot[1], -rot[3], -rot[2]

        elif idx in self.protocol_4:
            rot[1], rot[2], rot[3] = -rot[1], rot[3], rot[2]

        elif idx in self.protocol_5:
            # rot[0], rot[1], rot[2], rot[3] = rot[0], rot[3], rot[1], rot[2] # ori
            # rot[0], rot[1], rot[2], rot[3] = -rot[1], -rot[0], -rot[3], -rot[2] # rotate x
            # rot[0], rot[1], rot[2], rot[3] = -rot[2], rot[3], rot[0], -rot[1] # rotate y 180
            # rot[0], rot[1], rot[2], rot[3] = -rot[3], -rot[2], rot[1], rot[0] # rotate z 180
            rot[0], rot[1], rot[2], rot[3] = -rot[1], rot[2], rot[0], -rot[3]

        elif idx in self.left_hand:
            rot[1], rot[2], rot[3] = rot[1], rot[2], -rot[3]

        elif idx in self.right_hand:
            rot[1], rot[2], rot[3] = rot[1], rot[2], rot[3]

        return rot.tolist()

    def export_pcg(self, rot, msg, idx):

        #msg += "{0},{1},{2},{3},".format(rot[1], rot[2], rot[3], rot[0])
        if idx in self.protocol_0:
            msg += "{0},{1},{2},{3},".format(-rot[2], rot[3], -rot[1], rot[0])

        # two arms
        elif idx in self.protocol_1:
            msg += "{0},{1},{2},{3},".format(-rot[1], -rot[3], -rot[2], rot[0])
        elif idx in self.protocol_2:
            msg += "{0},{1},{2},{3},".format(rot[1], -rot[3], rot[2], rot[0])

        # two legs, corrected
        elif idx in self.protocol_3:
            msg += "{0},{1},{2},{3},".format(rot[2], rot[3], rot[1], rot[0])
        elif idx in self.protocol_4:
            msg += "{0},{1},{2},{3},".format(rot[2], rot[3], rot[1], rot[0])

        # global
        elif idx == 0 and idx in self.protocol_5:
            msg += "{0},{1},{2},{3},".format(rot[3], rot[1], -rot[2], rot[0])

        # two hands, corrected
        elif idx in self.left_hand:
            msg += "{0},{1},{2},{3},".format(-rot[1], rot[2], rot[3], rot[0])
        elif idx in self.right_hand:
            msg += "{0},{1},{2},{3},".format(rot[1], -rot[2], -rot[3], rot[0])

        else:
            print(idx)
            raise ValueError

        return msg

    def angle_limit(poses):
        return results

    def export(self, poses, trans, face=None, wrist='hand', protocol='gilly', frame_id=None):
        # poses = np.array(poses).reshape(-1, 3, 3)
        poses = np.array(poses).reshape(-1, 4)
        poses = torch.from_numpy(poses)

        #################################
        # option 1. hand wrist
        if wrist == 'hand':
            print('using ACR wrist')
            mat_rots = poses.numpy()
            length = len(mat_rots)
            assert length == 54, f'{length}' # 22 body + 32 hand
            mat_rots[20], mat_rots[21] = mat_rots[22], mat_rots[38]

            ##################################################
            # # mat_rots[1:] = self.pose_abs2rel(mat_rots[0][None][None], mat_rots[1:][None], 'left_wrist')
            # # mat_rots[1:] = self.pose_abs2rel(mat_rots[0][None][None], mat_rots[1:][None], 'right_wrist')
            # mat_rots = torch.from_numpy(mat_rots)
            # if mat_rots.shape[1:] == (3,3):
            #     rot_mat = mat_rots.reshape(-1, 3, 3)
            #     hom = torch.tensor([0, 0, 1], dtype=torch.float32,
            #                     device=mat_rots.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
            #     mat_rots = torch.cat([rot_mat, hom], dim=-1)

            # quat = self.rotation_matrix_to_quaternion(mat_rots)
            ##################################################
            quat = mat_rots

            # quat[22], quat[38] = quat[20], quat[21]
            # #quat[0][0],quat[0][1],quat[0][2],quat[0][3]=quat[0][0],quat[0][1],quat[0][2]-0.085,quat[0][3]
            # quat[0][0],quat[0][1],quat[0][2],quat[0][3]=quat[0][0],quat[0][1],quat[0][2]-0.085,quat[0][3]-0.02

            quat[15][0],quat[15][1],quat[15][2],quat[15][3]=quat[15][0],quat[15][1]-0.24,quat[15][2],quat[15][3]-0.16
            # quat[7][0],quat[7][1],quat[7][2],quat[7][3]=1.,0.,0.,0.
            # quat[8][0],quat[8][1],quat[8][2],quat[8][3]=1.,0.,0.,0.

            # quat[29][0],quat[29][1],quat[29][2],quat[29][3]=quat[29][0],quat[29][1],quat[29][2]-0.4,quat[29][3]
            # quat[26][0],quat[26][1],quat[26][2],quat[26][3]=quat[26][0],quat[26][1],quat[26][2]-0.08,quat[26][3]
            # quat[32][0],quat[32][1],quat[32][2],quat[32][3]=quat[32][0],quat[32][1],quat[32][2]-0.16,quat[32][3]

            # quat[45][0],quat[45][1],quat[45][2],quat[45][3]=quat[45][0],quat[45][1],quat[45][2]+0.4,quat[45][3]
            # quat[42][0],quat[42][1],quat[42][2],quat[42][3]=quat[42][0],quat[42][1],quat[42][2]+0.16,quat[42][3]
            # quat[48][0],quat[48][1],quat[48][2],quat[48][3]=quat[48][0],quat[48][1],quat[48][2]+0.24,quat[48][3]

            # quat[30][0],quat[30][1],quat[30][2],quat[30][3]=quat[30][0],quat[30][1],quat[30][2]-0.16,quat[30][3]
            # quat[31][0],quat[31][1],quat[31][2],quat[31][3]=quat[31][0],quat[31][1],quat[31][2]-0.08,quat[31][3]

            # quat[47][0],quat[47][1],quat[47][2],quat[47][3]=quat[47][0],quat[47][1],quat[47][2],quat[47][3]+0.3
            # #quat[46][0],quat[46][1],quat[46][2],quat[46][3]=quat[46][0],quat[46][1]+0.3,quat[46][2],quat[46][3]+0.3
            # quat[46][0],quat[46][1],quat[46][2],quat[46][3]=quat[46][0],quat[46][1]-0.3,quat[46][2],quat[46][3]+0.15
            assert quat.shape == (length, 4),f'{quat.shape}'

        # quat = quat.numpy()
        # print(quat)

        ###############################
        # add buffer
        if protocol == 'gilly':
            results_dict = {}
            if frame_id == 0:
                self.init_translation = np.array([-trans[1], -trans[2], trans[0]])
                translation  = np.array([0,0,0])
            else:
                translation = np.array([-trans[1], -trans[2], trans[0]]) - self.init_translation
                #translation = np.array([-trans[1], -trans[2], trans[0]])

            rotations = {}
            for idx in self.used_id:
                rot = quat[idx]
                msg = self.export_gilly_new(rot, self.bone_name_from_index[idx], idx)
                rotations[self.bone_name_from_index[idx]] = msg

            results_dict['translation'] = translation.tolist()
            results_dict['rotations'] = rotations
            self.buffer_json.append(results_dict)

        else:
            raise NotImplementedError
        
        
        ########################################## FACE
        if face is not None:
            pass
        else:
            results_dict['morphs'] = {
                "Brow_Lowerer": 0, # only use them as an integeral for now. Could be further split to l and r.
                #"Brow_Lowerer": -(face_data['eyebrow_updown_l'] + face_data['eyebrow_updown_r']), # only use them as an integeral for now. Could be further split to l and r.
                "EH": 0, 
                "AAA": 0, 
                "AHH":  0, 
                "IEE": 0,
                "UUU": 0,
                "OHH": 0,
                "Lip_Corner_Puller": 0,
                "Blink": 0
            }

        return quat

    #############
    # transform
    def pose_abs2rel(self, global_pose, body_pose, abs_joint = 'head'):
        ''' change absolute pose to relative pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        '''
        assert len(body_pose.shape) == 4
        assert len(global_pose.shape) == 4
        if abs_joint == 'head':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == 'neck':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == 'right_wrist':
            # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
            # -> right elbow -> right wrist
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == 'left_wrist':
            # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
            # -> Left elbow -> Left wrist
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(
                f'pose_abs2rel does not support: {abs_joint}')

        batch_size = global_pose.shape[0]
        assert batch_size == 1
        dtype = global_pose.dtype
        #device = global_pose.device
        full_pose = np.concatenate([global_pose, body_pose], axis=1)
        rel_rot_mat = np.eye(
            3, #device=device,
            dtype=dtype)[None]#.repeat(batch_size, 1, 1)
        for idx in kin_chain[1:]:
            rel_rot_mat = np.matmul(full_pose[:, idx], rel_rot_mat)

        # This contains the absolute pose of the parent
        abs_parent_pose = rel_rot_mat#.detach()
        # Let's assume that in the input this specific joint is predicted as an absolute value
        abs_joint_pose = body_pose[:, kin_chain[0] - 1]
        # abs_head = parents(abs_neck) * rel_head ==> rel_head = abs_neck.T * abs_head
        rel_joint_pose = np.matmul(
            abs_parent_pose.reshape(-1, 3, 3).transpose(0, 2, 1),
            abs_joint_pose.reshape(-1, 3, 3))
        # Replace the new relative pose
        body_pose[:, kin_chain[0] - 1, :, :] = rel_joint_pose

        return body_pose

    def angle_axis_to_quaternion(self,
        angle_axis: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
    ) -> torch.Tensor:
        r"""Convert an angle axis to a quaternion.
        The quaternion vector has components in (x, y, z, w) or (w, x, y, z) format.
        Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
        Args:
            angle_axis: tensor with angle axis in radians.
            order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.
        Return:
            tensor with quaternion.
        Shape:
            - Input: :math:`(*, 3)` where `*` means, any number of dimensions
            - Output: :math:`(*, 4)`
        Example:
            >>> angle_axis = torch.tensor((0., 1., 0.))
            >>> angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
            tensor([0.8776, 0.0000, 0.4794, 0.0000])
        """
        if not torch.is_tensor(angle_axis):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(angle_axis)}")

        if not angle_axis.shape[-1] == 3:
            raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {angle_axis.shape}")

        if not torch.jit.is_scripting():
            if order.name not in QuaternionCoeffOrder.__members__.keys():
                raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

        if order == QuaternionCoeffOrder.XYZW:
            print(
                "`XYZW` quaternion coefficient order is deprecated and",
                " will be removed after > 0.6. ",
                "Please use `QuaternionCoeffOrder.WXYZ` instead."
            )

        # unpack input and compute conversion
        a0: torch.Tensor = angle_axis[..., 0:1]
        a1: torch.Tensor = angle_axis[..., 1:2]
        a2: torch.Tensor = angle_axis[..., 2:3]
        theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

        theta: torch.Tensor = torch.sqrt(theta_squared)
        half_theta: torch.Tensor = theta * 0.5

        mask: torch.Tensor = theta_squared > 0.0
        ones: torch.Tensor = torch.ones_like(half_theta)

        k_neg: torch.Tensor = 0.5 * ones
        k_pos: torch.Tensor = torch.sin(half_theta) / theta
        k: torch.Tensor = torch.where(mask, k_pos, k_neg)
        w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

        quaternion: torch.Tensor = torch.zeros(
            size=(*angle_axis.shape[:-1], 4), dtype=angle_axis.dtype, device=angle_axis.device
        )
        if order == QuaternionCoeffOrder.XYZW:
            quaternion[..., 0:1] = a0 * k
            quaternion[..., 1:2] = a1 * k
            quaternion[..., 2:3] = a2 * k
            quaternion[..., 3:4] = w
        else:
            quaternion[..., 1:2] = a0 * k
            quaternion[..., 2:3] = a1 * k
            quaternion[..., 3:4] = a2 * k
            quaternion[..., 0:1] = w
        return quaternion

    def rotation_matrix_to_quaternion(self, rotation_matrix, eps=1e-6):
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

    def quat2mat(self, quat):
        """
        This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

        Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                                2], norm_quat[:,
                                                                            3]

        batch_size = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
            w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
            w2 - x2 - y2 + z2
        ],
                            dim=1).view(batch_size, 3, 3)
        return rotMat

    def batch_rodrigues(self, axisang):
        # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
        # axisang N x 3
        axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(axisang_norm, -1)
        axisang_normalized = torch.div(axisang, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
        rot_mat = self.quat2mat(quat)
        #rot_mat = rot_mat.view(rot_mat.shape[0], 9)
        return rot_mat



def ready_mano_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    #from mano.webuser.posemapper import posemap

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in [
            'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
            'betas', 'J'
    ]:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert (posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        #pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        #dd['v_posed'] = v_shaped + dd['posedirs'].dot(pose_map_res)
    else:
        pass
        #pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        #dd_add = dd['posedirs'].dot(pose_map_res)
        #dd['v_posed'] = dd['v_template'] + dd_add

    return dd



#######face
class Feature():
    def __init__(self, threshold=0.15, alpha=0.2, hard_factor=0.15, decay=0.001, max_feature_updates=0):
        self.median = remedian()
        self.min = None
        self.max = None
        self.hard_min = None
        self.hard_max = None
        self.threshold = threshold
        self.alpha = alpha
        self.hard_factor = hard_factor
        self.decay = decay
        self.last = 0
        self.current_median = 0
        self.update_count = 0
        self.max_feature_updates = max_feature_updates
        self.first_seen = -1
        self.updating = True

    def update(self, x, now=0):
        if self.max_feature_updates > 0:
            if self.first_seen == -1:
                self.first_seen = now
        new = self.update_state(x, now=now)
        filtered = self.last * self.alpha + new * (1 - self.alpha)
        #print('filtered', filtered, self.last, self.alpha, new, x)
        self.last = filtered
        return filtered

    def update_state(self, x, now=0):
        updating = self.updating and (self.max_feature_updates == 0 or now - self.first_seen < self.max_feature_updates)
        if updating:
            self.median + x
            self.current_median = self.median.median()
        else:
            self.updating = False
        median = self.current_median

        if self.min is None:
            if x < median and (median - x) / median > self.threshold:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
            return 0
        else:
            if x < self.min:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
        if self.max is None:
            if x > median and (x - median) / median > self.threshold:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                #print('!!!!!!!!!!!!!!!!!!!!', x, median, self.threshold)
                return 1
            return 0
        else:
            if x > self.max:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                #print('????????????????????')
                return 1

        if updating:
            if self.min < self.hard_min:
                self.min = self.hard_min * self.decay + self.min * (1 - self.decay)
            if self.max > self.hard_max:
                self.max = self.hard_max * self.decay + self.max * (1 - self.decay)

        if x < median:
            return - (1 - (x - self.min) / (median - self.min))
        elif x > median:
            return (x - median) / (self.max - median)

        return 0

class FeatureExtractor():
    def __init__(self, max_feature_updates=0):
        self.eye_l = Feature(max_feature_updates=max_feature_updates)
        self.eye_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_l = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_r = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_open = Feature(max_feature_updates=max_feature_updates)
        self.mouth_wide = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_A = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_E = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_I = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_O = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_U = Feature(threshold=0.02, max_feature_updates=max_feature_updates)

    def update(self, poses):

        features = {}
        now = time.perf_counter()

        # norm_distance_x = np.mean([pts[0, 0] - pts[16, 0], pts[1, 0] - pts[15, 0]])
        # norm_distance_y = np.mean([pts[27, 1] - pts[28, 1], pts[28, 1] - pts[29, 1], pts[29, 1] - pts[30, 1]])

        # a1, f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        # f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        # features["eye_l"] = self.eye_l.update(f, now)

        # a2, f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        # f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        # features["eye_r"] = self.eye_r.update(f, now)

        # if full:
        #     a3, _ = self.align_points(pts[0], pts[16], [])
        #     a4, _ = self.align_points(pts[31], pts[35], [])
        #     norm_angle = np.mean(list(map(np.rad2deg, [a1, a2, a3, a4])))

        #     a, f_pts = self.align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
        #     features["eyebrow_steepness_l"] = self.eyebrow_steepness_l.update(-np.rad2deg(a) - norm_angle, now)
        #     f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
        #     features["eyebrow_quirk_l"] = self.eyebrow_quirk_l.update(f, now)

        #     a, f_pts = self.align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
        #     features["eyebrow_steepness_r"] = self.eyebrow_steepness_r.update(np.rad2deg(a) - norm_angle, now)
        #     f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
        #     features["eyebrow_quirk_r"] = self.eyebrow_quirk_r.update(f, now)
        # else:
        #     features["eyebrow_steepness_l"] = 0.
        #     features["eyebrow_steepness_r"] = 0.
        #     features["eyebrow_quirk_l"] = 0.
        #     features["eyebrow_quirk_r"] = 0.

        # f = (np.mean([pts[22, 1], pts[26, 1]]) - pts[27, 1]) / norm_distance_y
        # features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f, now)

        # f = (np.mean([pts[17, 1], pts[21, 1]]) - pts[27, 1]) / norm_distance_y
        # features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f, now)

        # upper_mouth_line = np.mean([pts[49, 1], pts[50, 1], pts[51, 1]])
        # center_line = np.mean([pts[50, 0], pts[60, 0], pts[27, 0], pts[30, 0], pts[64, 0], pts[55, 0]])

        # f = (upper_mouth_line - pts[62, 1]) / norm_distance_y
        # features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f, now)
        # if full:
        #     f = abs(center_line - pts[62, 0]) / norm_distance_x
        #     features["mouth_corner_inout_l"] = self.mouth_corner_inout_l.update(f, now)
        # else:
        #     features["mouth_corner_inout_l"] = 0.

        # f = (upper_mouth_line - pts[58, 1]) / norm_distance_y
        # features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f, now)
        # if full:
        #     f = abs(center_line - pts[58, 0]) / norm_distance_x
        #     features["mouth_corner_inout_r"] = self.mouth_corner_inout_r.update(f, now)
        # else:
        #     features["mouth_corner_inout_r"] = 0.

        ###
        # mouth open and wide
        # f = abs(np.mean(pts[[59,60,61], 1], axis=0) - np.mean(pts[[63,64,65], 1], axis=0)) / norm_distance_y
        # features["mouth_open"] = self.mouth_open.update(f, now)

        # f = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x
        # features["mouth_wide"] = self.mouth_wide.update(f, now)

        if poses is not None:
            eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio, x_angle, y_angle, z_angle, ratioA, ratioE, ratioI, ratioO, ratioU, eyeopen_l, eyeopen_r, new_mouth_wide, mouth_open, l_brow, r_brow = poses

            # ?????????????????????????????????????????????????????????????????????????
            # features["mouth_A"] = self.mouth_A.update(ratioA, now)
            # features["mouth_E"] = self.mouth_E.update(ratioE, now)
            # features["mouth_I"] = self.mouth_I.update(ratioI, now)
            # features["mouth_O"] = self.mouth_O.update(ratioO, now)
            # features["mouth_U"] = self.mouth_U.update(ratioU, now)
            features["mouth_A"] = ratioA
            features["mouth_E"] = ratioE
            features["mouth_I"] = ratioI
            features["mouth_O"] = ratioO
            features["mouth_U"] = ratioU
            features["eye_l"] = self.eye_l.update(eyeopen_l, now)
            features["eye_r"] = self.eye_r.update(eyeopen_r, now)
            #print("returned new_mouth_wide", new_mouth_wide)
            features["mouth_open"] = mouth_open
            features["mouth_wide"] = new_mouth_wide # self.mouth_wide.update(new_mouth_wide, now)
            #print('updated new_mouth_wide', features["mouth_wide"])
            features["eyebrow_updown_l"] = l_brow #self.eyebrow_updown_l.update(l_brow, now)
            features["eyebrow_updown_r"] = r_brow #self.eyebrow_updown_r.update(r_brow, now)

            #print('ratioA, ratioE, ratioI, ratioO, ratioU', ratioA, ratioE, ratioI, ratioO, ratioU)
        else:
            raise ValueError
            features["mouth_A"] = 0.
            features["mouth_E"] = 0.
            features["mouth_I"] = 0.
            features["mouth_O"] = 0.
            features["mouth_U"] = 0.

        return features

class FaceInfo():
    def __init__(self, id, tracker):
        self.id = id
        self.frame_count = -1
        self.tracker = tracker

        self.reset()
        self.alive = False
        self.coord = None

        if self.tracker.max_feature_updates > 0:
            self.features = FeatureExtractor(self.tracker.max_feature_updates)

    def reset(self):
        self.alive = False
        self.eye_state = None
        self.rotation = None
        self.translation = None
        self.eye_blink = None
        self.bbox = None

        if self.tracker.max_feature_updates < 1:
            self.features = FeatureExtractor(0)
        self.current_features = {}
        self.fail_count = 0

    def update(self, ff):
        self.alive = True
        self.current_features = self.features.update(ff)
        self.eye_blink = []
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_r"]), 1))
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_l"]), 1))



        

class Face_Smoother:

    def __init__(self):

        self.left_eye = "Left_eye_skinJnt"
        self.right_eye = "Right_eye_skinJnt"
    
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = np.array([value])
        else:
            self.smooth[name] = np.insert(arr=self.smooth[name], obj=0, values=value)
            #print(name, self.smooth[name], value, type(value[0]), type(self.smooth[name]), self.smooth[name].dtype)
            if self.smooth[name].size > length:
                self.smooth[name] = np.delete(self.smooth[name], self.smooth[name].size-1, 0)

        sum = 0
        for val in self.smooth[name]:
            sum += val

        return sum / self.smooth[name].size

    def process_poses(self, face_data, blink):
        final_dict = {
                "Brow_Lowerer": float(self.smooth_value('brow', 2, [min(max(-(face_data['eyebrow_updown_l'] + face_data['eyebrow_updown_r']), -0.5), 1.0)])),
                "EH": float(self.smooth_value('EH', 2, [min(max(face_data['mouth_A'], 0.0), 1.0)])), 
                "AAA": float(self.smooth_value('AHH', 2, [min(max(face_data['mouth_I'], 0.0), 1.0)])), 
                "AHH":  float(self.smooth_value('AHH', 2, [min(max(face_data['mouth_open'], 0.0), 1.0)])), 
                "IEE": float(self.smooth_value('IEE', 2, [min(max(face_data['mouth_E'], 0.0), 1.0)])),
                "UUU": float(self.smooth_value('UUU', 3, [min(max(-face_data['mouth_wide'], 0.0), 1.0)])),
                "OHH": float(self.smooth_value('OHH', 2, [min(max(face_data['mouth_U'], 0.0), 1.0)])),
                "Lip_Corner_Puller": float(self.smooth_value('Lip_Corner_Puller', 3, [min(max(face_data['mouth_wide'], 0.0), 0.5)])),
                "Blink": 1-np.mean(blink) # no single blink for Gilly
        }
        print('brow:', final_dict['Brow_Lowerer'])

        return final_dict

def main():
    face_count = 0
    left_data = ready_mano_arguments('../../Tencent_gilly_json_files/models/MANO_LEFT.pkl')
    left_hands_mean = left_data['hands_mean']
    left_hands_mean = left_hands_mean.copy()

    right_data = ready_mano_arguments('../../Tencent_gilly_json_files/models/MANO_RIGHT.pkl')
    right_hands_mean = right_data['hands_mean']
    right_hands_mean = right_hands_mean.copy()

    frame_id = 0
    face_valid = False
    face_smoother = Face_Smoother()
    frame_id = 0
    results_dict = {}
    pcg = PCG(face_smoother=face_smoother)

    # GT
    data = np.load('../mydiffusion_zeggs/beat_zyx_quat.npy')
    all_trans = np.load('../mydiffusion_zeggs/beat_zyx_transl.npy')

    smplx_index = np.array([1,2,3,4,6,8,   10,11,12,13,13,  14,15,16,  19,20,21,  24,56,26,  29,30,31,  33,34,35,  37,38,39,40,40,  41,42,43,  46,47,48,  51,52,53,  56,57,58,  60,61,62,  64,65,66,68,  70,71,72,74]) - 1
    corr_index =  np.array([0,3,6,9,12,15, 14,17,19,38,21,  42,43,44,  48,49,50,  45,46,47,  39,40,41,  51,52,53,  13,16,18,22,20,  26,27,28,  32,33,34,  29,30,31,  23,24,25,  35,36,37,  2,5,8,11,     1,4,7,10])
    
    # pred
    for x, tmp_trans in zip(data, all_trans):
        poses = np.zeros((54,4))
        if x.shape[0] == 75:
            poses[corr_index] = x[smplx_index]
        else:
            poses[:22,:] = x[:22,:]
            poses[22] = x[20]
            poses[38] = x[21]
            poses[23:38,:] = x[25:40,:]
            poses[39:54,:] = x[40:55,:]

    
        poses = list(poses.reshape(-1))

        out = None
        if out is None:
            face_count += 1
                   # 
        tmp_trans *= 100.0
        tmp_trans[0],tmp_trans[1],tmp_trans[2] = tmp_trans[0],-tmp_trans[2],-tmp_trans[1]
        new_trans = -tmp_trans[1], -tmp_trans[2], tmp_trans[0]
        #new_trans = 0, tmp_trans[0], 0
        _ = pcg.export(poses, new_trans, out, wrist='hand', protocol='gilly', frame_id=frame_id)
        #_ = time.time()
        print(f'****** sending frame: {frame_id} ********')
        frame_id += 1

    pcg.export_json("../../Tencent_gilly_json_files/gtnew.json")
    


# def main(data,all_trans):
#     face_count = 0
#     left_data = ready_mano_arguments('../../Tencent_gilly_json_files/models/MANO_LEFT.pkl')
#     left_hands_mean = left_data['hands_mean']
#     left_hands_mean = left_hands_mean.copy()

#     right_data = ready_mano_arguments('../../Tencent_gilly_json_files/models/MANO_RIGHT.pkl')
#     right_hands_mean = right_data['hands_mean']
#     right_hands_mean = right_hands_mean.copy()

#     frame_id = 0
#     face_valid = False
#     face_smoother = Face_Smoother()
#     frame_id = 0
#     results_dict = {}
#     pcg = PCG(face_smoother=face_smoother)

#     # GT

#     # smplx_index = np.array([1,2,3,4,6,8,   10,11,12,13,13,  14,15,16,  19,20,21,  24,56,26,  29,30,31,  33,34,35,  37,38,39,40,40,  41,42,43,  46,47,48,  51,52,53,  56,57,58,  60,61,62,  64,65,66,68,  70,71,72,74]) - 1
#     # corr_index =  np.array([0,3,6,9,12,15, 14,17,19,38,21,  42,43,44,  48,49,50,  45,46,47,  39,40,41,  51,52,53,  13,16,18,22,20,  26,27,28,  32,33,34,  29,30,31,  23,24,25,  35,36,37,  2,5,8,11,     1,4,7,10])
    
#     # pred
#     for x, tmp_trans in zip(data, all_trans):
#         poses = np.zeros((54,4))
#         poses[:22,:] = x[:22,:]
#         poses[22] = x[20]
#         poses[38] = x[21]
#         poses[23:38,:] = x[25:40,:]
#         poses[39:54,:] = x[40:55,:]

    
#         poses = list(poses.reshape(-1))

#         out = None
#         if out is None:
#             face_count += 1
#                    # 
#         tmp_trans *= 100.0
#         tmp_trans[0],tmp_trans[1],tmp_trans[2] = tmp_trans[0],-tmp_trans[2],-tmp_trans[1]
#         new_trans = -tmp_trans[1], -tmp_trans[2], tmp_trans[0]
#         #new_trans = 0, tmp_trans[0], 0
#         _ = pcg.export(poses, new_trans, out, wrist='hand', protocol='gilly', frame_id=frame_id)
#         #_ = time.time()
#         print(f'****** sending frame: {frame_id} ********')
#         frame_id += 1

#     pcg.export_json("../../Tencent_gilly_json_files/gtnew.json")


def batch_rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)



def quat_to_rotmat(quat):
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

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
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

def quat2json(smplx_aa,transl):
    pred_pose_rotmat = batch_rodrigues(torch.tensor(all_smplx).cuda().reshape(-1,3))
    if pred_pose_rotmat.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)

    quat = rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,55,4)
    quat_numpy = quat.cpu().numpy().reshape(-1,55,4)
    quat_numpy = quat_numpy.reshape(-1,55,4)
    global_orientation = quat_numpy[:,0,:]
    def quaternion_rotation(quaternion, angle, axis):
        # 将角度转换为弧度
        angle_rad = np.radians(angle)

        # 计算旋转四元数
        rotation_quaternion = np.array([
            np.cos(angle_rad / 2),
            axis[0] * np.sin(angle_rad / 2),
            axis[1] * np.sin(angle_rad / 2),
            axis[2] * np.sin(angle_rad / 2)
        ])

        # 四元数乘法（旋转）
        temp = np.zeros(4)
        temp[0] = rotation_quaternion[0] * quaternion[0] - np.dot(rotation_quaternion[1:], quaternion[1:])
        temp[1:] = rotation_quaternion[0] * quaternion[1:] + quaternion[0] * rotation_quaternion[1:] + np.cross(rotation_quaternion[1:], quaternion[1:])

        return temp
    for i in range(len(global_orientation)):
        quaternion = global_orientation[i]  # 单位四元数，表示无旋转

        # 沿x轴旋转90度
        angle = 270
        axis = np.array([1, 0, 0])

        rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
        quat_numpy[i,0,:] = rotated_quaternion
    main(quat_numpy,transl)


def pose2json2(npz_file):
    npz_path = npz_file
    data=np.load(npz_path,allow_pickle=True)
    body = data['poses'][:,:22,:]#22
    hands = data['poses'][:,25:,:]#30
    jaw = data['poses'][:,22:25,:]
    all_smplx = np.concatenate((body,jaw,hands),axis=1)
    np.save("beat_zyx_transl.npy",data['trans'])
    pred_pose_rotmat = batch_rodrigues(torch.tensor(all_smplx).cuda().reshape(-1,3))
    if pred_pose_rotmat.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)

    quat = rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,55,4)

    quat_numpy = quat.cpu().numpy().reshape(-1,55,4)


    global_orientation = quat_numpy[:,0,:]
    def quaternion_rotation(quaternion, angle, axis):
        # 将角度转换为弧度
        angle_rad = np.radians(angle)

        # 计算旋转四元数
        rotation_quaternion = np.array([
            np.cos(angle_rad / 2),
            axis[0] * np.sin(angle_rad / 2),
            axis[1] * np.sin(angle_rad / 2),
            axis[2] * np.sin(angle_rad / 2)
        ])

        # 四元数乘法（旋转）
        temp = np.zeros(4)
        temp[0] = rotation_quaternion[0] * quaternion[0] - np.dot(rotation_quaternion[1:], quaternion[1:])
        temp[1:] = rotation_quaternion[0] * quaternion[1:] + quaternion[0] * rotation_quaternion[1:] + np.cross(rotation_quaternion[1:], quaternion[1:])

        return temp

    # 示例四元数
    if 'pharse' in npz_path:
        np.save("beat_zyx_quat.npy",quat_numpy)
    elif 'text' in npz_path:
        for i in range(len(global_orientation)):
            quaternion = global_orientation[i]  # 单位四元数，表示无旋转

            # 沿x轴旋转90度
            angle = 270
            axis = np.array([1, 0, 0])

            rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
            quat_numpy[i,0,:] = rotated_quaternion



        np.save("beat_zyx_quat.npy",quat_numpy)
    else:
        np.save("beat_zyx_quat.npy",quat_numpy)
    main()




# if __name__ == '__main__':
#     npz_path = "/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/demo/main/mydiffusion_zeggs/our_gen_text.npz"
#     data=np.load(npz_path,allow_pickle=True)



#     body = data['poses'][:,:22,:]#22
#     hands = data['poses'][:,25:,:]#30
#     jaw = data['poses'][:,22:25,:]
#     all_smplx = np.concatenate((body,jaw,hands),axis=1)
#     #np.save("beat_npy.npy",all_smplx)
#     np.save("beat_zyx_transl.npy",data['trans'])


#     pred_pose_rotmat = batch_rodrigues(torch.tensor(all_smplx).cuda().reshape(-1,3))



#     if pred_pose_rotmat.shape[1:] == (3,3):
#         hom_mat = torch.tensor([0, 0, 1]).float()
#         rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
#         batch_size, device = rot_mat.shape[0], rot_mat.device
#         hom_mat = hom_mat.view(1, 3, 1)
#         hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
#         hom_mat = hom_mat.to(device)
#         pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)

#     quat = rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,55,4)

#     quat_numpy = quat.cpu().numpy().reshape(-1,55,4)


#     global_orientation = quat_numpy[:,0,:]
#     def quaternion_rotation(quaternion, angle, axis):
#         # 将角度转换为弧度
#         angle_rad = np.radians(angle)

#         # 计算旋转四元数
#         rotation_quaternion = np.array([
#             np.cos(angle_rad / 2),
#             axis[0] * np.sin(angle_rad / 2),
#             axis[1] * np.sin(angle_rad / 2),
#             axis[2] * np.sin(angle_rad / 2)
#         ])

#         # 四元数乘法（旋转）
#         temp = np.zeros(4)
#         temp[0] = rotation_quaternion[0] * quaternion[0] - np.dot(rotation_quaternion[1:], quaternion[1:])
#         temp[1:] = rotation_quaternion[0] * quaternion[1:] + quaternion[0] * rotation_quaternion[1:] + np.cross(rotation_quaternion[1:], quaternion[1:])

#         return temp

#     # 示例四元数
#     if 'pharse' in npz_path:
#         np.save("beat_zyx_quat.npy",quat_numpy)
#     else:
#         for i in range(len(global_orientation)):
#             quaternion = global_orientation[i]  # 单位四元数，表示无旋转

#             # 沿x轴旋转90度
#             angle = 270
#             axis = np.array([1, 0, 0])

#             rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
#             quat_numpy[i,0,:] = rotated_quaternion



#         np.save("beat_zyx_quat.npy",quat_numpy)


if __name__ == '__main__':
    # npz_path = "/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/ExpGes_demo/demo/main/mydiffusion_zeggs/our_gen_text.npz"
    npz_path = "/apdcephfs/share_1290939/shaolihuang/ykcheng/SIGASIA_realtime_A2G/HoloGest/HoloGest_audio2gesture/main/mydiffusion_zeggs/beat2_our_rot.npz"
    data=np.load(npz_path,allow_pickle=True)



    body = data['poses'][:,:22,:]#22
    hands = data['poses'][:,25:,:]#30
    jaw = data['poses'][:,22:25,:]
    all_smplx = np.concatenate((body,jaw,hands),axis=1)
    #np.save("beat_npy.npy",all_smplx)

    # 定义旋转角度
    rotation_x = np.radians(270)  # 绕x轴旋转270度
    rotation_y = np.radians(0)  # 绕y轴旋转270度

    # 定义旋转矩阵
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_x), -np.sin(rotation_x)],
        [0, np.sin(rotation_x), np.cos(rotation_x)]
    ])

    rotation_matrix_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # 计算总旋转矩阵
    rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_x)

    # 定义位移向量
    translation = data['trans']  # 假设有1000帧，每帧有xyz三个维度

    # 计算新的位移向量
    new_translation = np.dot(translation, rotation_matrix.T) 
    print("new_translation shape:",new_translation.shape,"    data['trans'] shape:",data['trans'].shape)

    np.save("beat_zyx_transl.npy",new_translation)


    pred_pose_rotmat = batch_rodrigues(torch.tensor(all_smplx).cuda().reshape(-1,3))



    if pred_pose_rotmat.shape[1:] == (3,3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = pred_pose_rotmat.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        pred_pose_rotmat = torch.cat([rot_mat, hom_mat], dim=-1)

    quat = rotation_matrix_to_quaternion(pred_pose_rotmat).reshape(-1,55,4)

    quat_numpy = quat.cpu().numpy().reshape(-1,55,4)
    

    global_orientation = quat_numpy[:,0,:]
    def quaternion_rotation(quaternion, angle, axis):
        # 将角度转换为弧度
        angle_rad = np.radians(angle)

        # 计算旋转四元数
        rotation_quaternion = np.array([
            np.cos(angle_rad / 2),
            axis[0] * np.sin(angle_rad / 2),
            axis[1] * np.sin(angle_rad / 2),
            axis[2] * np.sin(angle_rad / 2)
        ])

        # 四元数乘法（旋转）
        temp = np.zeros(4)
        temp[0] = rotation_quaternion[0] * quaternion[0] - np.dot(rotation_quaternion[1:], quaternion[1:])
        temp[1:] = rotation_quaternion[0] * quaternion[1:] + quaternion[0] * rotation_quaternion[1:] + np.cross(rotation_quaternion[1:], quaternion[1:])

        return temp

    # 示例四元数
    if 'pharse' in npz_path:
        np.save("beat_zyx_quat.npy",quat_numpy)
    else:
        for i in range(len(global_orientation)):
            quaternion = global_orientation[i]  # 单位四元数，表示无旋转

            # 沿x轴旋转270度
            angle = 270
            axis = np.array([1, 0, 0])

            rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
            quat_numpy[i,0,:] = rotated_quaternion

            # # 沿y轴旋转90度
            # angle = 90
            # axis = np.array([0, 1, 0])

            # rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
            # quat_numpy[i,0,:] = rotated_quaternion
        global_orientation = quat_numpy[:,0,:]
        for i in range(len(global_orientation)):
            quaternion = global_orientation[i]  # 单位四元数，表示无旋转
            # 沿y轴旋转90度
            angle = 0
            axis = np.array([0, 1, 0])

            rotated_quaternion = quaternion_rotation(quaternion, angle, axis)
            quat_numpy[i,0,:] = rotated_quaternion


        

        np.save("beat_zyx_quat.npy",quat_numpy)




        
