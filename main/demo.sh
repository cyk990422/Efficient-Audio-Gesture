cd ./hologest

# running command
python holgest_infer.py --config=./configs/HoloGest.yml --no_cuda 0 --gpu 0 --model_path './hologest_model.pt' --audiowavlm_path './017_Relaxed_1_mirror_x_1_0_retarget2mocap.wav'  --speaker_style 'wayne'

# 
cd ./MotionPrior
# Flip the generated results to the position where the Y-axis points upward, conforming to the AMASS dataset and the 100-style standard.
python rot4mp.py
# Trajectory re-optimization
python difftraj_entry.py 

python add_finger.py
# output is the ../beat2_our_rot_cvt_allposw.npz

