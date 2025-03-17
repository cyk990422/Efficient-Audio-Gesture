import sys
import pdb

assert sys.version_info >= (3, 7)

BEAT_MOTION_TYPES = [
    "00_nogesture",
    "01_beat_align",
    "02_deictic_l",
    "03_deictic_m",
    "04_deictic_h",
    "05_iconic_l",
    "06_iconic_m",
    "07_iconic_h",
    "08_metaphoric_l",
    "09_metaphoric_m",
    "10_metaphoric_h",
    "habit",
    "need_cut",
]


# 75 joints
BEAT_JOINTS = {
    "Hips": [6, 6],  # dim, end_idx, Hips is root and contains global translation
    "Spine": [3, 9],
    "Spine1": [3, 12],
    "Spine2": [3, 15],
    "Spine3": [3, 18],
    "Neck": [3, 21],
    "Neck1": [3, 24],
    "Head": [3, 27],
    "HeadEnd": [3, 30],
    "RShoulder": [3, 33],
    "RArm": [3, 36],
    "RArm1": [3, 39],
    "RHand": [3, 42],
    "RHandM1": [3, 45],
    "RHandM2": [3, 48],
    "RHandM3": [3, 51],
    "RHandM4": [3, 54],
    "RHandR": [3, 57],
    "RHandR1": [3, 60],
    "RHandR2": [3, 63],
    "RHandR3": [3, 66],
    "RHandR4": [3, 69],
    "RHandP": [3, 72],
    "RHandP1": [3, 75],
    "RHandP2": [3, 78],
    "RHandP3": [3, 81],
    "RHandP4": [3, 84],
    "RHandI": [3, 87],
    "RHandI1": [3, 90],
    "RHandI2": [3, 93],
    "RHandI3": [3, 96],
    "RHandI4": [3, 99],
    "RHandT1": [3, 102],
    "RHandT2": [3, 105],
    "RHandT3": [3, 108],
    "RHandT4": [3, 111],
    "LShoulder": [3, 114],
    "LArm": [3, 117],
    "LArm1": [3, 120],
    "LHand": [3, 123],
    "LHandM1": [3, 126],
    "LHandM2": [3, 129],
    "LHandM3": [3, 132],
    "LHandM4": [3, 135],
    "LHandR": [3, 138],
    "LHandR1": [3, 141],
    "LHandR2": [3, 144],
    "LHandR3": [3, 147],
    "LHandR4": [3, 150],
    "LHandP": [3, 153],
    "LHandP1": [3, 156],
    "LHandP2": [3, 159],
    "LHandP3": [3, 162],
    "LHandP4": [3, 165],
    "LHandI": [3, 168],
    "LHandI1": [3, 171],
    "LHandI2": [3, 174],
    "LHandI3": [3, 177],
    "LHandI4": [3, 180],
    "LHandT1": [3, 183],
    "LHandT2": [3, 186],
    "LHandT3": [3, 189],
    "LHandT4": [3, 192],
    "RUpLeg": [3, 195],
    "RLeg": [3, 198],
    "RFoot": [3, 201],
    "RFootF": [3, 204],
    "RToeBase": [3, 207],
    "RToeBaseEnd": [3, 210],
    "LUpLeg": [3, 213],
    "LLeg": [3, 216],
    "LFoot": [3, 219],
    "LFootF": [3, 222],
    "LToeBase": [3, 225],
    "LToeBaseEnd": [3, 228],
}

# 75 joints, full BEAT skeleton, also include global position
BEAT_75 = {
    "Hips": 6,
    "Spine": 3,
    "Spine1": 3,
    "Spine2": 3,
    "Spine3": 3,
    "Neck": 3,
    "Neck1": 3,
    "Head": 3,
    "HeadEnd": 3,
    "RShoulder": 3,
    "RArm": 3,
    "RArm1": 3,
    "RHand": 3,
    "RHandM1": 3,
    "RHandM2": 3,
    "RHandM3": 3,
    "RHandM4": 3,
    "RHandR": 3,
    "RHandR1": 3,
    "RHandR2": 3,
    "RHandR3": 3,
    "RHandR4": 3,
    "RHandP": 3,
    "RHandP1": 3,
    "RHandP2": 3,
    "RHandP3": 3,
    "RHandP4": 3,
    "RHandI": 3,
    "RHandI1": 3,
    "RHandI2": 3,
    "RHandI3": 3,
    "RHandI4": 3,
    "RHandT1": 3,
    "RHandT2": 3,
    "RHandT3": 3,
    "RHandT4": 3,
    "LShoulder": 3,
    "LArm": 3,
    "LArm1": 3,
    "LHand": 3,
    "LHandM1": 3,
    "LHandM2": 3,
    "LHandM3": 3,
    "LHandM4": 3,
    "LHandR": 3,
    "LHandR1": 3,
    "LHandR2": 3,
    "LHandR3": 3,
    "LHandR4": 3,
    "LHandP": 3,
    "LHandP1": 3,
    "LHandP2": 3,
    "LHandP3": 3,
    "LHandP4": 3,
    "LHandI": 3,
    "LHandI1": 3,
    "LHandI2": 3,
    "LHandI3": 3,
    "LHandI4": 3,
    "LHandT1": 3,
    "LHandT2": 3,
    "LHandT3": 3,
    "LHandT4": 3,
    "RUpLeg": 3,
    "RLeg": 3,
    "RFoot": 3,
    "RFootF": 3,
    "RToeBase": 3,
    "RToeBaseEnd": 3,
    "LUpLeg": 3,
    "LLeg": 3,
    "LFoot": 3,
    "LFootF": 3,
    "LToeBase": 3,
    "LToeBaseEnd": 3,
}

BEAT_JOINTS.keys() == BEAT_75.keys()

# yapf: disable
BEAT_75_KINTREE = [
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15,
    12, 17, 18, 19, 20, 17, 22, 23, 24, 25, 12, 27, 28, 29, 30, 27, 32,
    33, 34, 4, 36, 37, 38, 39, 40, 41, 42, 39, 44, 45, 46, 47, 44, 49,
    50, 51, 52, 39, 54, 55, 56, 57, 54, 59, 60, 61, 0, 63, 64, 65, 66,
    67, 0, 69, 70, 71, 72, 73
]
# yapf: enable

BEAT_47 = {  # 47 joints, 141-D
    "Spine": 3,  # useful joint name, dim
    "Neck": 3,
    "Neck1": 3,
    "RShoulder": 3,
    "RArm": 3,
    "RArm1": 3,
    "RHand": 3,
    "RHandM1": 3,
    "RHandM2": 3,
    "RHandM3": 3,
    "RHandR": 3,
    "RHandR1": 3,
    "RHandR2": 3,
    "RHandR3": 3,
    "RHandP": 3,
    "RHandP1": 3,
    "RHandP2": 3,
    "RHandP3": 3,
    "RHandI": 3,
    "RHandI1": 3,
    "RHandI2": 3,
    "RHandI3": 3,
    "RHandT1": 3,
    "RHandT2": 3,
    "RHandT3": 3,
    "LShoulder": 3,
    "LArm": 3,
    "LArm1": 3,
    "LHand": 3,
    "LHandM1": 3,
    "LHandM2": 3,
    "LHandM3": 3,
    "LHandR": 3,
    "LHandR1": 3,
    "LHandR2": 3,
    "LHandR3": 3,
    "LHandP": 3,
    "LHandP1": 3,
    "LHandP2": 3,
    "LHandP3": 3,
    "LHandI": 3,
    "LHandI1": 3,
    "LHandI2": 3,
    "LHandI3": 3,
    "LHandT1": 3,
    "LHandT2": 3,
    "LHandT3": 3,
}
BEAT_16 = {
    "Hips": 3,
    "Spine": 3,
    "Spine1": 3,
    "Spine2": 3,
    "Spine3": 3,
    "Neck": 3,
    "Neck1": 3,
    "Head": 3,
    "RShoulder": 3,
    "RArm": 3,
    "RArm1": 3,
    "RHand": 3,
    "LShoulder": 3,
    "LArm": 3,
    "LArm1": 3,
    "LHand": 3,
}
BEAT_9 = {  # 9 joints, 27-D
    "Spine": 3,  # useful joint name, dim
    "Neck": 3,
    "Neck1": 3,
    "RShoulder": 3,
    "RArm": 3,
    "RArm1": 3,
    "LShoulder": 3,
    "LArm": 3,
    "LArm1": 3,
}

# 75 joints
NAME_MAPPER = {  # names in code to names in BVH
    "Hips": "Hips",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Spine3": "Spine3",
    "Neck": "Neck",
    "Neck1": "Neck1",
    "Head": "Head",
    "HeadEnd": "HeadEnd",
    "RShoulder": "RightShoulder",
    "RArm": "RightArm",
    "RArm1": "RightForeArm",
    "RHand": "RightHand",
    "RHandM1": "RightHandMiddle1",
    "RHandM2": "RightHandMiddle2",
    "RHandM3": "RightHandMiddle3",
    "RHandM4": "RightHandMiddle4",
    "RHandR": "RightHandRing",
    "RHandR1": "RightHandRing1",
    "RHandR2": "RightHandRing2",
    "RHandR3": "RightHandRing3",
    "RHandR4": "RightHandRing4",
    "RHandP": "RightHandPinky",
    "RHandP1": "RightHandPinky1",
    "RHandP2": "RightHandPinky2",
    "RHandP3": "RightHandPinky3",
    "RHandP4": "RightHandPinky4",
    "RHandI": "RightHandIndex",
    "RHandI1": "RightHandIndex1",
    "RHandI2": "RightHandIndex2",
    "RHandI3": "RightHandIndex3",
    "RHandI4": "RightHandIndex4",
    "RHandT1": "RightHandThumb1",
    "RHandT2": "RightHandThumb2",
    "RHandT3": "RightHandThumb3",
    "RHandT4": "RightHandThumb4",
    "LShoulder": "LeftShoulder",
    "LArm": "LeftArm",
    "LArm1": "LeftForeArm",
    "LHand": "LeftHand",
    "LHandM1": "LeftHandMiddle1",
    "LHandM2": "LeftHandMiddle2",
    "LHandM3": "LeftHandMiddle3",
    "LHandM4": "LeftHandMiddle4",
    "LHandR": "LeftHandRing",
    "LHandR1": "LeftHandRing1",
    "LHandR2": "LeftHandRing2",
    "LHandR3": "LeftHandRing3",
    "LHandR4": "LeftHandRing4",
    "LHandP": "LeftHandPinky",
    "LHandP1": "LeftHandPinky1",
    "LHandP2": "LeftHandPinky2",
    "LHandP3": "LeftHandPinky3",
    "LHandP4": "LeftHandPinky4",
    "LHandI": "LeftHandIndex",
    "LHandI1": "LeftHandIndex1",
    "LHandI2": "LeftHandIndex2",
    "LHandI3": "LeftHandIndex3",
    "LHandI4": "LeftHandIndex4",
    "LHandT1": "LeftHandThumb1",
    "LHandT2": "LeftHandThumb2",
    "LHandT3": "LeftHandThumb3",
    "LHandT4": "LeftHandThumb4",
    "RUpLeg": "RightUpLeg",
    "RLeg": "RightLeg",
    "RFoot": "RightFoot",
    "RFootF": "RightForeFoot",
    "RToeBase": "RightToeBase",
    "RToeBaseEnd": "RightToeBaseEnd",
    "LUpLeg": "LeftUpLeg",
    "LLeg": "LeftLeg",
    "LFoot": "LeftFoot",
    "LFootF": "LeftForeFoot",
    "LToeBase": "LeftToeBase",
    "LToeBaseEnd": "LeftToeBaseEnd",
}

NAME_MAPPER_INVERSE = {v: k for k, v in NAME_MAPPER.items()}
BEAT_JOINTS.keys() == NAME_MAPPER.keys()


# yapf: disable
speaker_names = [
    "wayne", "scott", "solomon", "lawrence", "stewart",
    "carla", "sophie", "catherine", "miranda", "kieks",
    "nidal", "zhao", "lu", "zhang", "carlos",
    "jorge", "itoi", "daiki", "jaime", "li",
    "ayana", "luqi", "hailing", "kexin", "goto",
    "reamey", "yingqing", "tiffnay", "hanieh", "katya",
]
recording_type = {
    '0': 'English Speech',
    '1': 'English Conversation',
    '2': 'Chinese Speech',
    '3': 'Chinese Conversation',
    '4': 'Spanish Speech',
    '5': 'Spanish Conversation',
    '6': 'Japanese Speech',
    '7': 'Japanese Conversation'
}
FOUR_HOUR_SPEAKERS = "1, 2, 3, 4, 6, 7, 8, 9, 11, 21"  # 10 in total
ONE_HOUR_SPEAKERS = "5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30"  # 20 in total
# yapf: enable

# yapf: disable
split_rule_EN = {
    # 4h speakers x 10
    FOUR_HOUR_SPEAKERS: {
        # 48+40+100=188mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16",
            "0_17_17", "0_18_18", "0_19_19", "0_20_20", "0_21_21", "0_22_22", "0_23_23", "0_24_24",
            "0_25_25", "0_26_26", "0_27_27", "0_28_28", "0_29_29", "0_30_30", "0_31_31", "0_32_32",
            "0_33_33", "0_34_34", "0_35_35", "0_36_36", "0_37_37", "0_38_38", "0_39_39", "0_40_40",
            "0_41_41", "0_42_42", "0_43_43", "0_44_44", "0_45_45", "0_46_46", "0_47_47", "0_48_48",
            "0_49_49", "0_50_50", "0_51_51", "0_52_52", "0_53_53", "0_54_54", "0_55_55", "0_56_56",  # neutral
            "0_66_66", "0_67_67", "0_68_68", "0_69_69", "0_70_70", "0_71_71",  # happy
            "0_74_74", "0_75_75", "0_76_76", "0_77_77", "0_78_78", "0_79_79",  # anger
            "0_82_82", "0_83_83", "0_84_84", "0_85_85",  # sad
            "0_88_88", "0_89_89", "0_90_90", "0_91_91", "0_92_92", "0_93_93",  # contempt
            "0_96_96", "0_97_97", "0_98_98", "0_99_99", "0_100_100", "0_101_101",  # suprise
            "0_104_104", "0_105_105", "0_106_106", "0_107_107", "0_108_108", "0_109_109",  # fear
            "0_112_112", "0_113_113", "0_114_114", "0_115_115", "0_116_116", "0_117_117",  # disgust
            "1_2_2", "1_3_3", "1_4_4", "1_5_5", "1_6_6", "1_7_7", "1_8_8", "1_9_9", "1_10_10", "1_11_11",  # conversation, neutral
        ],
        # 8+7+10=25mins each
        "val": [
            "0_57_57", "0_58_58", "0_59_59", "0_60_60", "0_61_61", "0_62_62", "0_63_63", "0_64_64",  # neutral
            "0_72_72", "0_80_80", "0_86_86", "0_94_94", "0_102_102", "0_110_110", "0_118_118",  # emotional
            "1_12_12",  # conversation
        ],
        # 8+7+10=25mins each
        "test": [
            "0_1_1", "0_2_2", "0_3_3", "0_4_4", "0_5_5", "0_6_6", "0_7_7", "0_8_8",  # neutral
            "0_65_65", "0_73_73", "0_81_81", "0_87_87", "0_95_95", "0_103_103", "0_111_111",  # emotional
            "1_1_1",  # conversation
        ],
    },

    # 1h speakers x 20
    ONE_HOUR_SPEAKERS: {
        # 8+7+20=35mins each
        "train": [
            "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16", \
            "0_66_66", "0_74_74", "0_82_82", "0_88_88", "0_96_96", "0_104_104", "0_112_112", "0_118_118", \
            "1_2_2", "1_3_3",
            "1_0_0", "1_4_4",  # for speaker 29 only
        ],
        # 4+3.5+5 = 12.5mins each
        # 0_65_a and 0_65_b denote the frist and second half of sequence 0_65_65
        "val": [
            "0_5_5", "0_6_6", "0_7_7", "0_8_8",  \
            "0_65_b", "0_73_b", "0_81_b", "0_87_b", "0_95_b", "0_103_b", "0_111_b", \
            "1_1_b",
        ],
        # 4+3.5+5 = 12.5mins each
        "test": [
            "0_1_1", "0_2_2", "0_3_3", "0_4_4", \
            "0_65_a", "0_73_a", "0_81_a", "0_87_a", "0_95_a", "0_103_a", "0_111_a", \
            "1_1_a",
        ],
    },
}
# missing sequences: S9: 2~8 (test); S21: 1~8(test); S15: 6 (val).
# yapf: enable
