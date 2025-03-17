import numpy as np

from zeggs import ZEGGS_75 as JoI
from zeggs import ZEGGS
from scipy.spatial.transform import Rotation
from MyBVH import load_bvh_data,write_bvh_data

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

bvh_fn='/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/DiffuseStyleGesture/main/mydiffusion_zeggs/sample_dir/20230626_153537_smoothing_SG_minibatch_640_[0, 1, 0, 0, 0, 0]_123456.bvh'
with open(bvh_fn, "r") as fp_bvh:
    lines = fp_bvh.readlines()
rot_data = []
for i, line_data in enumerate(lines, start=1):
    if i >= 464:
        data = np.fromstring(line_data, dtype=float, sep=" ")
        JoI_rot = []
        for name, d in JoI.items():
            start_idx = ZEGGS[name][1] - d
            end_idx = ZEGGS[name][1]
            cur_joint_rot = data[start_idx:end_idx]
            if name == "Hips":
                JoI_rot.append(cur_joint_rot[-3:])
            else:
                JoI_rot.append(cur_joint_rot)
        JoI_rot = np.concatenate(JoI_rot, axis=0)
        rot_data.append(JoI_rot)
rot_angles = np.stack(rot_data, axis=0)
rot_angles = rot_angles.reshape((len(rot_angles), -1, 3))
rot_angles = rot_angles.astype(np.float32)




zyx_euler_angles=rot_angles

'''
zyx_euler_angles[:,9,0]-=np.ones(zyx_euler_angles.shape[0])*90
zyx_euler_angles[:,35,0]+=np.ones(zyx_euler_angles.shape[0])*90
'''


xyz_euler_angles = []

# 将 ZYX 顺序的欧拉角逐个转换为 XYZ 顺序的欧拉角
for i in range(zyx_euler_angles.shape[0]):
    xyz_euler_angles_item = []
    
    for j in range(zyx_euler_angles.shape[1]):
        # 使用 ZYX 顺序的欧拉角创建一个 Rotation 对象
        '''
        if j==9:
            rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)
            xyz_euler_angle = rot_zyx.as_euler("ZYX", degrees=True)
            xyz_euler_angle[0]*=1
            xyz_euler_angle[1]*=1
            xyz_euler_angle[2]*=1
            xyz_euler_angles_item.append(xyz_euler_angle)
            continue
        elif j==10:
            rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)
            xyz_euler_angle = rot_zyx.as_euler("XYZ", degrees=True)
            xyz_euler_angle[0]*=1
            xyz_euler_angle[1]*=1
            xyz_euler_angle[2]*=1
            xyz_euler_angles_item.append(xyz_euler_angle)
            continue
        elif j==11:
            rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)
            xyz_euler_angle = rot_zyx.as_euler("ZYX", degrees=True)
            xyz_euler_angles_item.append(xyz_euler_angle)
            continue
        elif j==12:
            rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)
            xyz_euler_angle = rot_zyx.as_euler("ZYX", degrees=True)
            xyz_euler_angles_item.append(xyz_euler_angle)
            continue
        '''
        '''
        if j==10 or j==11:
            rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)
            xyz_euler_angle = rot_zyx.as_euler("ZYX", degrees=True)
            xyz_euler_angle[0]*=1
            xyz_euler_angle[1]*=1
            xyz_euler_angle[2]*=1
            xyz_euler_angles_item.append(xyz_euler_angle)
            continue
        '''
        rot_zyx = Rotation.from_euler("ZYX", zyx_euler_angles[i, j], degrees=True)

        # 将 Rotation 对象转换为 XYZ 顺序的欧拉角
        xyz_euler_angle = rot_zyx.as_euler("XYZ", degrees=True)
        xyz_euler_angles_item.append(xyz_euler_angle)

    xyz_euler_angles.append(xyz_euler_angles_item)

# 将结果列表转换成 NumPy 数组
xyz_euler_angles = np.array(xyz_euler_angles)
smplx_index = np.array([0,1,3,4,5,7,   9,10,11,12,12,   21,22,23,  25,26,27,  29,43,31,  17,18,19,  13,14,15,  35,36,37,38,38,  47,48,49,  51,52,53,  55,56,57,  43,44,45,  39,40,41,  61,62,64,64,  68,69,71,71])
corr_index =  np.array([0,3,6,9,12,15, 14,17,19,38,21,  42,43,44,  48,49,50,  45,46,47,  39,40,41,  51,52,53,  13,16,18,22,20,  26,27,28,  32,33,34,  29,30,31,  23,24,25,  35,36,37,  2,5,8,11,     1,4,7,10])
smplx_index_new = np.array([0,68,61,1,69,62,3,71,64,4,71,64,5,35,9,7,36,10,37,11,38,12,38,43,44,45,47,48,49,55,56,57,51,52,53,39,40,41,12,17,18,19,21,22,23,29,43,31,25,26,27,13,14,15])
np.savez("xyz_euler_pred.npz",rot_angles_new=xyz_euler_angles[:,smplx_index_new,:])
print("shape:",xyz_euler_angles[:,smplx_index_new,:].shape)

print("rot_angles:")
print(rot_angles[:2,35,:])

'''
xyz_euler_angles[:,9,2]-=75
#xyz_euler_angles[:,9,1]-=20
xyz_euler_angles[:,9,0]-=60

xyz_euler_angles[:,35,2]+=90
xyz_euler_angles[:,35,0]-=60
'''


'''

shoulder_x=xyz_euler_angles[:,9,0].copy()
shoulder_y=xyz_euler_angles[:,9,1].copy()
shoulder_z=xyz_euler_angles[:,9,2].copy()

shoulder_x1=xyz_euler_angles[:,35,0].copy()
shoulder_y1=xyz_euler_angles[:,35,1].copy()
shoulder_z1=xyz_euler_angles[:,35,2].copy()


xyz_euler_angles[:,9,0]=shoulder_x
xyz_euler_angles[:,9,1]=shoulder_z
xyz_euler_angles[:,9,2]=shoulder_y*-1

xyz_euler_angles[:,35,0]=shoulder_x1
xyz_euler_angles[:,35,1]=shoulder_z1
xyz_euler_angles[:,35,2]=shoulder_y1*-1+50

'''
'''
beat_index_cor=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,67,68,69,70,71,73,74]
zegg_index_cor=[0,1,2,3,4,5,6,7,8,9,10,11,12,21,22,23,24,25,26,27,28,29,30,31,32,17,18,19,20,13,14,15,16,35,36,37,38,47,48,49,50,51,52,53,54,55,56,57,58,43,44,45,46,39,40,41,42,61,62,63,64,65,68,69,70,71,72]
#beat_index_cor=[0,1,2,3,4,5,6,7,8,63,64,65,67,68,69,70,71,73,74,9,10,11,12,36,37,38,39]
#zegg_index_cor=[0,1,2,3,4,5,6,7,8,61,62,63,64,65,68,69,70,71,72,9,10,11,12,35,36,37,38]
#beat_index_cor=[0,1,2,3,4,5,6,7,8,63,64,65,67,68,69,70,71,73,74,9,10,11,12,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,32,33,34,35]
#zegg_index_cor=[0,1,2,3,4,5,6,7,8,61,62,63,64,65,68,69,70,71,72,9,10,11,12,21,22,23,24,25,26,27,28,29,30,31,32,17,18,19,20,13,14,15,16]
eular_pred_zeros=np.zeros(xyz_euler_angles.shape)
eular_pred_zeros[:,beat_index_cor,:]=xyz_euler_angles[:,zegg_index_cor,:]
eular_pred=eular_pred_zeros
'''
'''
eular_pred=xyz_euler_angles
print("eular_pred shape:",eular_pred.shape)
info = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_genea/examples/2_scott_1_1_1.bvh")
euler_orders=info["euler_orders"][: 75]

#info = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/DiffuseStyleGesture/main/mydiffusion_zeggs/sample_dir/20230626_153537_smoothing_SG_minibatch_640_[0, 1, 0, 0, 0, 0]_123456.bvh")
#pose_pred_npz=info["rot_angles"][:200]
#euler_orders=info["euler_orders"][: 75]

print("euler_orders:",euler_orders)
pred_pose_rotmat=euler2mat(eular_pred, euler_orders).astype(np.float32)#34,47,3,3
np.savez("pred_rotmat_new1.npz",pred_pose_rotmat=pred_pose_rotmat)
'''

info = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/DiffuseStyleGesture/main/mydiffusion_zeggs/sample_dir/20230626_153537_smoothing_SG_minibatch_640_[0, 1, 0, 0, 0, 0]_123456.bvh")
info1 = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/datasets/beat_cache/beat_4english_15_141test/bvh_full/1_wayne_0_2_2.bvh")


'''
print("info:")
print(info["joint_names"])
print(info["offsets"])
print(info["parents"])
print(info["euler_orders"])
print(info["framerate"])
print(info["global_pos"])


print("info1:")
print(info1["joint_names"])
print(info1["offsets"])
print(info1["parents"])
print(info1["euler_orders"])
print(info1["framerate"])
print(info1["global_pos"])


print("offsets shape:",info["offsets"].shape)
'''

#print("eular_pred:",eular_pred[:2,36,:])
#print("oreders shape:",np.array(info1["euler_orders"]).shape)
#print(info1["euler_orders"])
eular_pred=info["rot_angles"]*0
eular_pred[:,9,0]+=np.ones(eular_pred.shape[0])*90
eular_pred[:,35,0]-=np.ones(eular_pred.shape[0])*90
data = write_bvh_data(
    "3.bvh",
    joint_names=info["joint_names"],
    offsets=info["offsets"],
    skeleton_tree=info["parents"],
    euler_orders=info['euler_orders'],
    framerate=info["framerate"],
    motion=info["rot_angles"],
    global_trans=info['global_pos'],
    with_endsite=False,
)

eular_pred=xyz_euler_angles
#eular_pred[:,9,2]+=np.ones(eular_pred.shape[0])*90
#eular_pred[:,35,2]-=np.ones(eular_pred.shape[0])*90

pred_pose_rotmat=euler2mat(eular_pred, info1['euler_orders']).astype(np.float32)#34,47,3,3
np.savez("pred_rotmat_new1.npz",pred_pose_rotmat=pred_pose_rotmat)