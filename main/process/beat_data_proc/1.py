bone_names = [
        "Hips",#0                       'Hips'0             
        "Spine",#1                      'Spine'1
        "Spine1",#2                     'Spine1'2
        "Spine2",#3                     'Spine2'3
        "Spine3",#4                     'Spine3'4
        "Neck",#5                       'Neck'5
        "Neck1",#6                      'Neck1'6
        "Head",#7                       'Head'7
        "HeadEnd",#8                    'HeadEnd'8
        "RightShoulder",#9              'RShoulder'9
        "RightArm",#10                  'RArm'10
        "RightForeArm",#11              'RArm1'11
        "RightHand",#12                 'RHand'12
        "RightHandThumb1",#13           'RHandM1'21
        "RightHandThumb2",#14           'RHandM2'22
        "RightHandThumb3",#15           'RHandM3'23
        "RightHandThumb4",#16           'RHandM4'24
        "RightHandIndex1",#17           'RHandR'none
        "RightHandIndex2",#18           'RHandR1'25
        "RightHandIndex3",#19           'RHandR2'26
        "RightHandIndex4",#20           'RHandR3'27
        "RightHandMiddle1",#21          'RHandR4'28
        "RightHandMiddle2",#22          'RHandP'none
        "RightHandMiddle3",#23          'RHandP1'29   
        "RightHandMiddle4",#24          'RHandP2'30
        "RightHandRing1",#25            'RHandP3'31
        "RightHandRing2",#26            'RHandP4'32
        "RightHandRing3",#27            'RHandI'none
        "RightHandRing4",#28            'RHandI1'17
        "RightHandPinky1",#29           'RHandI2'18
        "RightHandPinky2",#30           'RHandI3'19
        "RightHandPinky3",#31           'RHandI4'20
        "RightHandPinky4",#32           'RHandT1'13
        "RightForeArmEnd",#33           'RHandT2'14
        "RightArmEnd",#34               'RHandT3'15
        "LeftShoulder",#35              'RHandT4'16
        "LeftArm",#36                   'LShoulder'35
        "LeftForeArm",#37               'LArm'36
        "LeftHand",#38                  'LArm1'37
        "LeftHandThumb1",#39            'LHand'38
        "LeftHandThumb2",#40            'LHandM1'47
        "LeftHandThumb3",#41            'LHandM2'48
        "LeftHandThumb4",#42            'LHandM3'49
        "LeftHandIndex1",#43            'LHandM4'50
        "LeftHandIndex2",#44            'LHandR'none
        "LeftHandIndex3",#45            'LHandR1'51
        "LeftHandIndex4",#46            'LHandR2'52
        "LeftHandMiddle1",#47           'LHandR3'53
        "LeftHandMiddle2",#48           'LHandR4'54
        "LeftHandMiddle3",#49           'LHandP'none
        "LeftHandMiddle4",#50           'LHandP1'55
        "LeftHandRing1",#51             'LHandP2'56
        "LeftHandRing2",#52             'LHandP3'57
        "LeftHandRing3",#53             'LHandP4'58
        "LeftHandRing4",#54             'LHandI'none
        "LeftHandPinky1",#55            'LHandI1'43
        "LeftHandPinky2",#56            'LHandI2'44
        "LeftHandPinky3",#57            'LHandI3'45
        "LeftHandPinky4",#58            'LHandI4'46
        "LeftForeArmEnd",#59            'LHandT1'39
        "LeftArmEnd",#60                'LHandT2'40
        "RightUpLeg",#61                'LHandT3'41
        "RightLeg",#62                  'LHandT4'42
        "RightFoot",#63                 'RUpLeg'61
        "RightToeBase",#64              'RLeg'62
        "RightToeBaseEnd",#65           'RFoot'63
        "RightLegEnd",#66               'RFootF'none
        "RightUpLegEnd",#67             'RToeBase'64
        "LeftUpLeg",#68                 'RToeBaseEnd'65
        "LeftLeg",#69                   'LUpLeg'68
        "LeftFoot",#70                  'LLeg'69
        "LeftToeBase",#71               'LFoot'70
        "LeftToeBaseEnd",#72            'LFootF'none
        "LeftLegEnd",#73                'LToeBase'71
        "LeftUpLegEnd"#74               'LToeBaseEnd'72
    ]

python sample.py --config=/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/configs/DiffuseStyleGesture_beat_vae.yml  --no_cuda 0 --gpu 0 --model_path '/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new4/DiffuseStyleGesture/main/mydiffusion_zeggs/beats_1_resume/model000450000.pt' --audiowavlm_path '/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_english_v0.2.1/10/10_kieks_0_3_3.wav' --max_len 0

python sample.py --config=./configs/DiffuseStyleGesture.yml --no_cuda 0 --gpu 0 --model_path '/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/ori/DiffuseStyleGesture/main/mydiffusion_zeggs/11.pt' --audiowavlm_path '/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/22_luqi_2_49_56_cut_90s.wav' --max_len 0

[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]

beat_47=[1,5,6,9,10,11,12,13,14,15,17,18,19,20,22,23,24,25,27,28,29,30,32,33,34,36,37,38,39,40,41,42,44,45,46,47,49,50,51,52,54,55,56,57,59,60,61]


[1,5,6,9,10,11,12,13,14,15,17,18,19,20,22,23,24,25,27,28,29,30,32,33,34,36,37,38,39,40,41,42,44,45,46,47,49,50,51,52,54,55,56,57,59,60,61]
    "beat_141" : {
            'Spine':       3 ,#1
            'Neck':        3 ,#5
            'Neck1':       3 ,#6
            'RShoulder':   3 ,#9 
            'RArm':        3 ,#10
            'RArm1':       3 ,#11
            'RHand':       3 ,#12    
            'RHandM1':     3 ,#13
            'RHandM2':     3 ,#14
            'RHandM3':     3 ,#15
            'RHandR':      3 ,#17
            'RHandR1':     3 ,#18
            'RHandR2':     3 ,#19
            'RHandR3':     3 ,#20
            'RHandP':      3 ,#22
            'RHandP1':     3 ,#23
            'RHandP2':     3 ,#24
            'RHandP3':     3 ,#25
            'RHandI':      3 ,#27
            'RHandI1':     3 ,#28
            'RHandI2':     3 ,#29
            'RHandI3':     3 ,#30
            'RHandT1':     3 ,#32
            'RHandT2':     3 ,#33
            'RHandT3':     3 ,#34
            'LShoulder':   3 , #36
            'LArm':        3 ,#37
            'LArm1':       3 ,#38
            'LHand':       3 ,   #39 
            'LHandM1':     3 ,#40
            'LHandM2':     3 ,#41
            'LHandM3':     3 ,#42
            'LHandR':      3 ,#44
            'LHandR1':     3 ,#45
            'LHandR2':     3 ,#46
            'LHandR3':     3 ,#47
            'LHandP':      3 ,#49
            'LHandP1':     3 ,#50
            'LHandP2':     3 ,#51
            'LHandP3':     3 ,#52
            'LHandI':      3 ,#54
            'LHandI1':     3 ,#55
            'LHandI2':     3 ,#56
            'LHandI3':     3 ,#57
            'LHandT1':     3 ,#59
            'LHandT2':     3 ,#60
            'LHandT3':     3 ,#61
        },