import mathutils

def parse_bvh_line(line):
    parts = line.strip().split()
    joint_name = parts[0]
    zyx_euler = tuple(map(float, parts[1:]))
    return joint_name, zyx_euler

def convert_euler_zyx_to_xyz(zyx_euler):
    zyx_matrix = mathutils.Euler(zyx_euler, 'ZYX').to_matrix()
    return zyx_matrix.to_euler('XYZ')

def process_bvh_line(line):
    joint_name, zyx_euler = parse_bvh_line(line)
    xyz_euler = convert_euler_zyx_to_xyz(zyx_euler)
    return f"{joint_name} {' '.join(map(str, xyz_euler))}"

def adjust_joint_hierarchy(line, joint_mapping):
    joint_name = line.strip().split()[1]
    new_joint_name = joint_mapping.get(joint_name, joint_name)
    return line.replace(joint_name, new_joint_name)

def convert_bvh_file(input_file, output_file, joint_mapping):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if "JOINT" in line or "ROOT" in line:
                outfile.write(adjust_joint_hierarchy(line, joint_mapping))
            elif "MOTION" in line:
                outfile.write(line)
                break
            else:
                outfile.write(process_bvh_line(line))
        outfile.writelines(infile.readlines())

input_bvh = "/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/codes/audio2pose/dataloaders/beat_data_proc/20230626_153537_smoothing_SG_minibatch_640_[0, 1, 0, 0, 0, 0]_123456.bvh"  # 替换为B骨架的BVH文件名
output_bvh = "/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/codes/audio2pose/dataloaders/beat_data_proc/1_wayne_0_2_2.bvh"  # 替换为您想要的A骨架BVH文件名
joint_mapping = {
    # 在这里添加B骨架关节到A骨架关节的映射，例如：
    "B_joint_name": "A_joint_name"
}
convert_bvh_file(input_bvh, output_bvh, joint_mapping)