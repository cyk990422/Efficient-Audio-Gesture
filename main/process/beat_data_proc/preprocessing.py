import os
import platform
from os.path import join as pjoin
import sys
import time
import pdb
import glob
import json
import math
import shutil
from collections import Counter
import numpy as np
#from utils_io import load_h5_dataset, decode_str
from scipy.io.wavfile import write

from natsort import os_sorted
from MyBVH import load_bvh_data, select_joints, euler2mat
from utils_io import load_h5_dataset, save_dataset_into_h5

from beat_meta_info import (
    speaker_names,
    recording_type,
    FOUR_HOUR_SPEAKERS,
    ONE_HOUR_SPEAKERS,
    BEAT_MOTION_TYPES,
    split_rule_EN,
)
from beat_meta_info import BEAT_JOINTS, NAME_MAPPER_INVERSE, NAME_MAPPER
from beat_meta_info import BEAT_75 as JOI  # BEAT_9, BEAT_16, BEAT_47, BEAT_75


if platform.system() == "Windows":
    BEAT_DIR = r"E:\BEAT-dataset\datasets"
else:
    BEAT_DIR = "/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/datasets/beat"

AUDIO_SAMPLING_RATE = 16000
TARGET_FPS = 15  # 15


def _get_special_seq():
    val_seqs = split_rule_EN[ONE_HOUR_SPEAKERS]["val"]
    test_seqs = split_rule_EN[ONE_HOUR_SPEAKERS]["test"]
    special_val_seqs = []
    for e in val_seqs:
        if e.endswith(("a", "b")):
            parts = e.split("_")
            special_val_seqs.append(f"{parts[0]}_{parts[1]}_{parts[1]}")

    special_test_seqs = []
    for e in test_seqs:
        if e.endswith(("a", "b")):
            parts = e.split("_")
            special_test_seqs.append(f"{parts[0]}_{parts[1]}_{parts[1]}")
    assert special_val_seqs == special_test_seqs
    return special_val_seqs


def parse_beat_filename(fn_no_ext, key=None):
    # filename format: []_[]_[]_[]_[].wav
    # refer to their [repo](https://github.com/PantoMatrix/BEAT#introcution)
    parts = fn_no_ext.split("_")
    speaker_id = parts[0]
    speaker_name = parts[1]
    recording_type = parts[2]
    sentence_id_start = parts[3]
    sentence_id_end = parts[4]
    split_id = f"{recording_type}_{sentence_id_start}_{sentence_id_end}"
    assert speaker_names[int(speaker_id) - 1] == speaker_name, f"{speaker_names[int(speaker_id)-1]} vs {speaker_name}"
    assert sentence_id_start == sentence_id_end
    info_dict = {
        "speaker_id": speaker_id,  # p0
        "speaker_name": speaker_name,  # p1
        "recording_type": recording_type,  # p2
        "sentence_id": sentence_id_start,  # p3
        "split_id": split_id,
    }
    return info_dict if key is None else info_dict[key]


def myFloatToString(inputValue):
    # return f"{inputValue:.6f}"
    return (f"{inputValue:.6f}").rstrip("0").rstrip(".")


def load_wave_data(wav_fn, sampling_rate=AUDIO_SAMPLING_RATE, savename=None):
    import librosa

    assert os.path.isfile(wav_fn)

    signal, _ = librosa.load(wav_fn, sr=sampling_rate, dtype=np.float32)
    if savename is None:
        return signal
    else:
        # savename = wav_fn.replace(".wav", ".npy")
        np.save(savename, signal)


def batch_wav2npy(wav_dir, npy_dir):
    wav_fns = glob.glob(pjoin(wav_dir, "*", "*.wav"))
    for wav_fn in wav_fns:
        fn = os.path.basename(wav_fn)
        subfolder = os.path.basename(os.path.dirname(wav_fn))
        os.makedirs(pjoin(npy_dir, subfolder), exist_ok=True)
        save_fn = pjoin(npy_dir, subfolder, fn.replace(".wav", ".npy"))
        load_wave_data(wav_fn, save_fn)


def read_json_file(fn):
    with open(fn, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


# def write_json_file(fn, data):
#     with open(fn, "w", encoding="utf-8") as f:
#         json.dump(data, f)


def downsample_json_file(fn_in, target_fps, *, fn_out=None):

    data = read_json_file(fn_in)
    t_interval = data["frames"][1]["time"] - data["frames"][0]["time"]
    fps = int(np.round(1 / t_interval))
    assert fps == 60
    assert np.mod(fps, target_fps) == 0
    decimation_factor = int(fps / target_fps)

    data_new = {"BS_names": data["names"], "BS_FPS": target_fps}
    coeffs = []
    for idx, frame_data in enumerate(data["frames"]):
        if idx % decimation_factor == 0:
            coeffs.append(np.asarray(frame_data["weights"], dtype=np.float32))
            assert len(frame_data["rotation"]) == 0

    data_new["BS_coeffs"] = np.stack(coeffs, axis=0).astype(np.float32)
    if fn_out is not None:
        save_dataset_into_h5(fn_out, data_new, verbose=False)
    else:
        return data_new


def my_array2string(vec, format_func=myFloatToString):
    assert vec.ndim == 1
    vec_str = np.array2string(
        vec,
        max_line_width=np.inf,
        precision=6,
        suppress_small=False,
        separator=" ",
        formatter={"float_kind": format_func},
    )
    return vec_str.rsplit("]", 1)[0].split("[", 1)[1]


def is_float(str):
    try:
        float(str)
    except ValueError:
        return False
    return True


def downsample_bvh_file(
    bvh_fn, save_dir, train_test, *, JoI, target_fps=TARGET_FPS, save_full_bvh=False, save_upper_bvh=False
):
    if target_fps is None:
        target_fps = 120
    assert np.mod(120, target_fps) == 0
    decimation_factor = int(120 / target_fps)

    fn = os.path.basename(bvh_fn).rsplit(".", 1)[0]

    if save_full_bvh:
        fn_decimated_full = pjoin(save_dir, train_test, "bvh_full", f"{fn}.bvh")  # original data
        fp_decimated_full = open(fn_decimated_full, "w")

    if save_upper_bvh:
        # partial motion, not necessarily upper
        fn_decimated_upper = pjoin(save_dir, train_test, "bvh_upper", f"{fn}.bvh")  # lower body fixed
        fp_decimated_upper = open(fn_decimated_upper, "w")

    rot_data = []
    joint_names = []
    if "Hips" in JoI:
        global_pos = []
    with open(bvh_fn, "r") as fp_bvh:
        lines = fp_bvh.readlines()
    for i, line_data in enumerate(lines, start=1):
        if i <= 429:
            if save_upper_bvh:
                fp_decimated_upper.write(line_data)
            if save_full_bvh:
                fp_decimated_full.write(line_data)
        elif i == 430:
            # e.g. Frames: 6240
            try:
                n_frame_old = int(line_data[8:])
            except ValueError:
                if is_float(line_data[8:]):
                    n_frame_old = int(float(line_data[8:]))
                else:
                    print(f"Please check this text:\n{line_data}")
                    sys.exit(1)
                print(f"Please check this text:\n{line_data}")
                # pdb.set_trace()
            n_frame_new = n_frame_old // decimation_factor
            line_data_new = f"Frames: {n_frame_new}\n"
            if save_upper_bvh:
                fp_decimated_upper.write(line_data_new)
            if save_full_bvh:
                fp_decimated_full.write(line_data_new)
        elif i == 431:
            # e.g. Frame Time: 0.008333
            try:
                interval_old = float(line_data[11:])
            except ValueError:
                pdb.set_trace()
            fps = int(np.round(1 / interval_old))
            if fps != 120:
                return None
            interval_new = 1.0 / target_fps
            line_data_new = f"Frame Time: {interval_new:.6f}\n"
            if save_upper_bvh:
                fp_decimated_upper.write(line_data_new)
            if save_full_bvh:
                fp_decimated_full.write(line_data_new)
        elif i >= 432 and (i - 432) % decimation_factor == 0:
            # if i % decimation_factor == 0:
            if save_full_bvh:
                fp_decimated_full.write(line_data)

            data = np.fromstring(line_data, dtype=float, sep=" ")
            if save_upper_bvh:
                # upper_motion: only keep upper body motion, set the others to zeros
                upper_motion = np.zeros_like(data)

            # JoI_rot: Joint of Interest, only keep the rotation data of JoI
            JoI_rot = []
            for name, d in JoI.items():
                start_idx = BEAT_JOINTS[name][1] - d
                end_idx = BEAT_JOINTS[name][1]
                cur_joint_rot = data[start_idx:end_idx]
                if name == "Hips":
                    global_pos.append(cur_joint_rot[:3])
                    JoI_rot.append(cur_joint_rot[-3:])
                else:
                    JoI_rot.append(cur_joint_rot)
                if name not in joint_names:
                    joint_names.append(name)
                if save_upper_bvh:
                    upper_motion[start_idx:end_idx] = cur_joint_rot

            JoI_rot = np.concatenate(JoI_rot, axis=0)
            rot_data.append(JoI_rot)

            if save_upper_bvh:
                upper_motion_str = my_array2string(upper_motion, format_func=myFloatToString)
                fp_decimated_upper.write(f"{upper_motion_str}\n")

    fp_bvh.close()
    if save_upper_bvh:
        fp_decimated_upper.close()
    if save_full_bvh:
        fp_decimated_full.close()

    info = load_bvh_data(bvh_fn)
    data_dict = {}
    joint_names = [NAME_MAPPER[j] for j in joint_names]
    data_dict["joint_names"] = joint_names
    parents_new, offsets_new, _ = select_joints(
        joint_names, info["joint_names"], parents=info["parents"], offsets=info["offsets"]
    )

    data_dict["framerate"] = float(target_fps)
    data_dict["parents"] = parents_new
    data_dict["offsets"] = offsets_new
    data_dict["euler_orders"] = info["euler_orders"][: len(joint_names)]  # TODO: update this line

    rot_angles = np.stack(rot_data, axis=0)  # nFrame x 141/228/27
    rot_angles = rot_angles.reshape((len(rot_angles), -1, 3))
    data_dict["rot_angles"] = rot_angles.astype(np.float32)
    data_dict["global_pos"] = np.stack(global_pos, axis=0).astype(np.float32)
    data_dict["rot_mats"] = euler2mat(rot_angles, data_dict["euler_orders"]).astype(np.float32)
    return data_dict


def task_prep_train_val_test_data(
    data_dir=None,
    train_test="train",
    *,
    target_fps=TARGET_FPS,
    save_full_bvh=False,
    save_upper_bvh=False,
    overwrite=False,
):
    assert train_test in ["train", "test", "val", "special"]
    assert np.mod(60, target_fps) == 0
    print(f"target_fps: {target_fps}")

    if train_test == "special":
        special_seqs = _get_special_seq()

    if data_dir is None:
        data_dir = BEAT_DIR

    ori_data_path = pjoin(data_dir, "beat_english_v0.2.1")
    ori_data_path_ann = ori_data_path
    save_dir = pjoin(data_dir, f"beat_English_{target_fps}FPS_{len(JOI)}")

    os.makedirs(pjoin(save_dir, train_test), exist_ok=True)
    os.makedirs(pjoin(save_dir, train_test, "data"), exist_ok=True)
    os.makedirs(pjoin(save_dir, train_test, "annot"), exist_ok=True)

    if save_upper_bvh:
        os.makedirs(pjoin(save_dir, train_test, "bvh_upper"), exist_ok=True)
    if save_full_bvh:
        os.makedirs(pjoin(save_dir, train_test, "bvh_full"), exist_ok=True)

    for speaker in range(1, 31):
        filenames = os_sorted(glob.glob(pjoin(ori_data_path, f"{speaker}", "*.wav")))
        # pdb.set_trace()
        for fn in filenames:
            fn = os.path.basename(fn).rsplit(".", 1)[0]
            save_fn = pjoin(save_dir, train_test, "data", f"{fn}.h5")
            if os.path.isfile(save_fn) and not overwrite:
                print(f"exists {save_fn}")
                continue

            info = parse_beat_filename(fn)
            if train_test == "special":
                if info["speaker_id"] not in ONE_HOUR_SPEAKERS.split(", "):
                    continue
                if info["split_id"] not in special_seqs:
                    continue
            else:
                person_group = (
                    ONE_HOUR_SPEAKERS if info["speaker_id"] in ONE_HOUR_SPEAKERS.split(", ") else FOUR_HOUR_SPEAKERS
                )
                # if train_test != "train" and person_group == ONE_HOUR_SPEAKERS:
                #     # val/test splits for the one-hour speakers are special
                #     print
                #     raise NotImplementedError
                # else:
                if info["split_id"] not in split_rule_EN[person_group][train_test]:
                    print(
                        "{}_{}: {} does not belong to {}".format(
                            info["speaker_id"], info["speaker_name"], info["split_id"], train_test
                        )
                    )
                    # NOTE: val/test splits for the one-hour speakers are special. They are not processed in this function.
                    continue

            # print(info["split_id"])
            # print(person_group)
            # print(split_rule_EN[person_group][train_test])
            # pdb.set_trace()
            data_dict = {}

            # load data from wav file
            wav_fn = pjoin(ori_data_path, fn.split("_")[0], f"{fn}.wav")
            print(f"Processing {wav_fn}")
            if not os.path.isfile(wav_fn):
                pdb.set_trace()
            wave_data = load_wave_data(wav_fn=wav_fn)
            data_dict["wave16k"] = wave_data
            wave_dur = len(data_dict["wave16k"]) / AUDIO_SAMPLING_RATE

            bvh_fn = pjoin(ori_data_path, fn.split("_")[0], f"{fn}.bvh")
            motion_data = downsample_bvh_file(
                bvh_fn,
                save_dir,
                train_test,
                JoI=JOI,
                save_full_bvh=save_full_bvh,
                save_upper_bvh=save_upper_bvh,
            )
            if motion_data is None:
                print(f"Problematic data for\n\t{bvh_fn}")
                continue
            motion_dur = len(motion_data["rot_angles"]) / float(motion_data["framerate"])
            data_dict.update(motion_data)

            fn_in = pjoin(ori_data_path, fn.split("_")[0], f"{fn}.json")
            # fn_out = pjoin(save_dir, train_test, "facial52", f"{fn}.h5")
            facial_data = downsample_json_file(fn_in, target_fps, fn_out=None)
            facial_dur = len(facial_data["BS_coeffs"]) / float(facial_data["BS_FPS"])
            data_dict.update(facial_data)

            final_dur = min(wave_dur, motion_dur, facial_dur)
            assert wave_dur >= final_dur, "Audio should always be longer than motion/facial animation"
            if (
                not np.isclose(final_dur, wave_dur, atol=0.07)
                or not np.isclose(final_dur, motion_dur, atol=0.07)
                or not np.isclose(final_dur, facial_dur, atol=0.07)
            ):
                print(
                    f"    >>>> length comparisons: wave {wave_dur:.3f}, motion {motion_dur:.3f}, facial {facial_dur:.3f}"
                )
                time.sleep(0.5)

            wave_samples = int(np.round(final_dur * AUDIO_SAMPLING_RATE))
            motion_frames = int(np.round(final_dur * motion_data["framerate"]))
            facial_frames = int(np.round(final_dur * facial_data["BS_FPS"]))
            data_dict["wave16k"] = data_dict["wave16k"][:wave_samples]
            data_dict["rot_angles"] = data_dict["rot_angles"][:motion_frames]
            data_dict["BS_coeffs"] = data_dict["BS_coeffs"][:facial_frames]
            save_dataset_into_h5(save_fn, data_dict, verbose=False)

            try:
                shutil.copy(
                    pjoin(ori_data_path, fn.split("_")[0], f"{fn}.TextGrid"),
                    # pjoin(save_dir, train_test, "text", f"{fn}.TextGrid"),
                    pjoin(save_dir, train_test, "annot", f"{fn}.TextGrid"),
                )
            except:
                print(f"{fn}.TextGrid")
                pdb.set_trace()

            try:
                shutil.copy(
                    pjoin(ori_data_path_ann, fn.split("_")[0], f"{fn}.txt"),
                    # pjoin(save_dir, train_test, "sem", f"{fn}.txt"),
                    pjoin(save_dir, train_test, "annot", f"{fn}.txt"),
                )
            except:
                print(f"{fn}.txt")
                pdb.set_trace()

            try:
                shutil.copy(
                    pjoin(ori_data_path_ann, fn.split("_")[0], f"{fn}.csv"),
                    # pjoin(save_dir, train_test, "emo", f"{fn}.csv"),
                    pjoin(save_dir, train_test, "annot", f"{fn}.csv"),
                )
            except:
                print(f"{fn}.csv")
                pdb.set_trace()


def get_mu_std(data, axis=0):
    # calculate stats across samples/frames
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return mean, std


def calc_speaker_stats(filenames, data_type="wave", fps=None):

    beat_fps = {"wave": AUDIO_SAMPLING_RATE, "motion": TARGET_FPS, "facial": TARGET_FPS}
    beat_data_name = {"wave": "wave16k", "motion": "rot_angles", "facial": "BS_coeffs"}
    assert data_type in ["wave", "motion", "facial"]
    if fps is None:
        fps = beat_fps[data_type]

    data_all = []
    for fn in filenames:
        data_dict = load_h5_dataset(fn, verbose=False)
        data_all.append(data_dict[beat_data_name[data_type]])

    data_all = np.concatenate(data_all, axis=0)
    print(f">>>> shape of data_all: {data_all.shape}")
    mu, std = get_mu_std(data_all)
    total_time = len(data_all) / float(fps)  # in seconds
    return {"total_time": total_time, "mean": mu, "std": std}


def task_calc_per_speaker_stats(data_dir, train_test="train", overwrite=False):
    for s_id in range(1, 31):
        h5_dir = pjoin(data_dir, train_test)
        save_fn = pjoin(h5_dir, f"speaker_{s_id}_{train_test}_stats.h5")
        if os.path.isfile(save_fn) and not overwrite:
            tmp = load_h5_dataset(save_fn)
            print(os.path.basename(save_fn))
            print(f"[wave] mean: {tmp['wave_mean']}, std: {tmp['wave_std']}")
            print(f"[body] mean: {tmp['motion_mean']}, std: {tmp['motion_std']}")
            print(f"[face] mean: {tmp['facial_mean']}, std: {tmp['facial_std']}")
            continue

        speaker_stats = {}
        h5_files = glob.glob(pjoin(data_dir, train_test, "data", f"{s_id}_*_*_*_*.h5"))
        if len(h5_files) == 0:
            continue
        for dtype in ["wave", "motion", "facial"]:
            cur_stats = calc_speaker_stats(h5_files, data_type=dtype, fps=None)
            if "total_time" not in speaker_stats:
                speaker_stats["total_time"] = cur_stats["total_time"]
            else:
                if not np.isclose(speaker_stats["total_time"], cur_stats["total_time"], atol=0.07):
                    # 0.07 seconds, up to one frame for 15FPS data
                    print(f"total_time: {speaker_stats['total_time']} vs {cur_stats['total_time']} sec")
                    pdb.set_trace()
            speaker_stats[f"{dtype}_mean"] = cur_stats["mean"]
            speaker_stats[f"{dtype}_std"] = cur_stats["std"]

        save_dataset_into_h5(save_fn, speaker_stats, verbose=False)


def task_cmp_per_speaker_stats(data_dir, save_fn):
    info = {}
    for s_id in range(1, 31):
        for key in ["train", "val", "test"]:
            fn = pjoin(data_dir, key, f"speaker_{s_id}_{key}_stats.h5")
            tmp = load_h5_dataset(fn)
            for k in tmp:
                info[f"speaker_{s_id}_{key}_{k}"] = tmp[k]
    save_dataset_into_h5(save_fn, info)
    return info


def calc_accumulated_stats(file_list, data_type="wave", fps=None):
    # calculate global mean and std for average all speakers

    beat_fps = {"wave": AUDIO_SAMPLING_RATE, "motion": TARGET_FPS, "facial": TARGET_FPS}
    assert data_type in ["wave", "motion", "facial"]
    if fps is None:
        fps = beat_fps[data_type]

    duration_list = []
    mu_list = []
    std_list = []
    for fn in file_list:
        stats = load_h5_dataset(fn)
        duration_list.append(stats["total_time"])
        mu_list.append(stats[f"{data_type}_mean"])
        std_list.append(stats[f"{data_type}_std"])

    # calculate accumulated mu and std
    total_samples = 0
    total_time = 0.0
    new_mu = np.zeros_like(mu_list[0])
    new_std = np.zeros_like(std_list[0])
    for t, mu, std in zip(duration_list, mu_list, std_list):
        total_time += t
        n = t * fps
        total_samples += n
        new_mu += n * mu
        new_std += (mu**2 + std**2) * n

    new_mu /= total_samples
    new_std /= total_samples
    new_std -= new_mu**2
    new_std = np.sqrt(new_std)

    print(f"accumulated mean: {new_mu}, std: {new_std}")
    return {"total_time": total_time, "mean": new_mu, "std": new_std}


def task_calc_dataset_stats(data_dir, train_test, save_fn):
    print("Calculating dataset stats ...")
    dtypes = ["wave", "motion", "facial"]
    file_list = os_sorted(glob.glob(pjoin(data_dir, f"speaker_*_{train_test}_stats.h5")))

    stats = {}
    for dtype in dtypes:
        cur_stats = calc_accumulated_stats(file_list, data_type=dtype)
        if "total_time" in stats:
            assert np.isclose(stats["total_time"], cur_stats["total_time"])
        else:
            stats["total_time"] = cur_stats["total_time"]
        stats[f"{dtype}_mean"] = cur_stats["mean"]
        stats[f"{dtype}_std"] = cur_stats["std"]
    save_dataset_into_h5(save_fn, stats, verbose=False)


# def split_data(data_dir, val_test="val"):
#     # spilt data
#     folders = os.listdir(data_dir)
#     if not os.path.exists(data_dir.replace("train", val_test)):
#         os.mkdir(data_dir.replace("train", val_test))

#     endwith = []
#     for folder in folders:
#         os.makedirs(pjoin(data_dir.replace("train", val_test), folder), exist_ok=True)
#         endwith.append(os.listdir(pjoin(data_dir, folder))[500].split(".")[-1])

#     for idx in [2]:  # speaker ID
#         subset = (
#             split_rule_EN[ONE_HOUR_SPEAKERS][val_test]
#             if str(idx) in ONE_HOUR_SPEAKERS.split(",")
#             else split_rule_EN[FOUR_HOUR_SPEAKERS][val_test]
#         )
#         for file_id in subset:
#             for ide, folder in enumerate(folders):
#                 if ("b" not in file_id) and ("a" not in file_id):
#                     src_fn = pjoin(data_dir, folder, f"{idx}_{speaker_names[idx-1]}_{file_id}.{endwith[ide]}")
#                     dst_dir = pjoin({data_dir.replace("train", val_test)}, folder)
#                     try:
#                         shutil.move(src_fn, dst_dir)
#                     except:
#                         print("Error.")
#                         print(src_fn)
#                         pdb.set_trace()
#                 elif val_test == "val":
#                     # split the sentence into two parts, a for test, b for val
#                     #     so we only need to do it once
#                     source_fn = "{p3}_{p4}_{p5}_{p6}_{p7}.{p8}".format(
#                         p3=idx,
#                         p4=speaker_names[idx - 1],
#                         p5=file_id.split("_")[0],
#                         p6=file_id.split("_")[1],
#                         p7=file_id.split("_")[1],
#                         p8=endwith[ide],
#                     )
#                     source_fn = pjoin(data_dir, folder, source_fn)
#                     save_fn_a = "{p3}_{p4}_{p5}_{p6}_a.{p8}".format(
#                         p3=idx,
#                         p4=speaker_names[idx - 1],
#                         p5=file_id.split("_")[0],
#                         p6=file_id.split("_")[1],
#                         p8=endwith[ide],
#                     )
#                     save_fn_a = pjoin(data_dir.replace("train", "test"), folder, save_fn_a)
#                     save_fn_b = "{p3}_{p4}_{p5}_{p6}_b.{p8}".format(
#                         p3=idx,
#                         p4=speaker_names[idx - 1],
#                         p5=file_id.split("_")[0],
#                         p6=file_id.split("_")[1],
#                         p8=endwith[ide],
#                     )
#                     save_fn_b = pjoin(data_dir.replace("train", "val"), folder, save_fn_b)
#                     cut_sequence(
#                         source_fn=source_fn,
#                         save_fn_a=save_fn_a,
#                         save_fn_b=save_fn_b,
#                         file_id=file_id,
#                     )
#                     # TODO: move it to tmp dir


def task_prep_special_val_test_seqs(data_dir=None, target_fps=TARGET_FPS):

    from copy import deepcopy

    print("Deprecated.")
    print("Do not split original data. If you split, you also need to maintain the annotation files.")
    print("Instead, you can slice the part you need from the data files when you use them.")
    sys.exit(0)

    special_seqs = _get_special_seq()

    assert np.mod(60, target_fps) == 0
    print(f"target_fps: {target_fps}")

    if data_dir is None:
        data_dir = BEAT_DIR

    ori_data_path = pjoin(data_dir, "beat_english_v0.2.1")
    ori_data_path_ann = ori_data_path
    save_dir = pjoin(data_dir, f"beat_English_{target_fps}FPS_{len(JOI)}")

    for speaker in ONE_HOUR_SPEAKERS.split(", "):
        filenames = os_sorted(glob.glob(pjoin(save_dir, "special", "data", f"{speaker}_*.h5")))
        for fn in filenames:
            print(fn)
            fn = os.path.basename(fn).rsplit(".", 1)[0]
            src_fn = pjoin(save_dir, "special", "data", f"{fn}.h5")
            part_a_fn = pjoin(save_dir, "test", "data", f"{fn.rsplit('_', 1)[0]}_a.h5")
            part_b_fn = pjoin(save_dir, "val", "data", f"{fn.rsplit('_', 1)[0]}_b.h5")

            info = parse_beat_filename(fn)
            if info["split_id"] not in special_seqs:
                print(fn)
                print("    It does not belong to special sequences.")
                sys.exit(1)

            data = load_h5_dataset(src_fn, verbose=False)
            cut_point = (
                30 if info["recording_type"] == "0" else 300
            )  # in seconds, part-a is the 1st part, part-b is the 2nd part

            data_a = {
                "BS_FPS": data["BS_FPS"],
                "BS_names": data["BS_names"],
                "motion_FPS": data["motion_FPS"],
                "joint_names": data["joint_names"],
            }
            data_b = deepcopy(data_a)

            for k, fps in [("wave16k", 16000), ("BS_coeffs", TARGET_FPS), ("rotations", TARGET_FPS)]:
                data_a[k] = data[k][: cut_point * fps]
                data_b[k] = data[k][cut_point * fps :]
            save_dataset_into_h5(part_a_fn, data_a, verbose=False)
            save_dataset_into_h5(part_b_fn, data_b, verbose=False)
            # TODO: need to split annotation files too.


def inspect_joint_names(JoI):
    bvh_fn = os.path.join(BEAT_DIR, f"beat_English_30FPS_{len(JOI)}", "val/bvh_full/1_wayne_0_58_58.bvh")
    info_dict = load_bvh_data(bvh_fn)
    for n0, n1 in zip(info_dict["joint_names"], NAME_MAPPER_INVERSE.keys()):
        print(f"{n0}: {n1}/{NAME_MAPPER_INVERSE[n1]}")
    pdb.set_trace()
    for n in JoI.keys():
        print(f"{n} vs {NAME_MAPPER[n]}")
    pdb.set_trace()
    # for n0, n1 in zip(info_dict["joint_names"], JoI.keys()):
    #     print(f"{n0} vs {n1}")





def _parse_motion_label(line):
    # 0-types, 1-start, 2-end, 3-duration, 4-score, 5-keywords (that trigger current motion)
    # types:
    #   '00_nogesture', '01_beat_align',
    #   '02_deictic_l', '03_deictic_m', '04_deictic_h',
    #   '05_iconic_l', '06_iconic_m', '07_iconic_h',
    #   '08_metaphoric_l', '09_metaphoric_m', '10_metaphoric_h',
    #   'habit', 'need_cut',
    parts = line.split("\t")
    if len(parts) not in [5, 6]:
        parts = line.split()
        if len(parts) not in [5, 6]:
            pdb.set_trace()

    info = {
        "type": parts[0],
        "start": float(parts[1]),
        "end": float(parts[2]),
        "duration": float(parts[3]),
        "score": parts[4],
    }
    if len(parts) == 5:
        return info
    elif len(parts) == 6:
        info["keywords"] = parts[5]
        return info
    else:
        print(line)
        pdb.set_trace()
        raise ValueError()


def collect_motion_labels(files):
    all_annot = []
    seq_lengths = []
    for idx, fn in enumerate(files):
        if idx % 50 == 0:
            print(fn)
            print(f"{idx} ...")
        with open(fn, "r") as f:
            tmp = f.readlines()
            labels = [_parse_motion_label(line.strip()) for line in tmp if line.strip()]
            all_annot.extend(labels)
        # print(f"{len(all_annot)} annotations")
        seq_lengths.append(all_annot[-1]["end"])

    all_annot_dict = {"type": [], "start": [], "end": [], "duration": [], "score": [], "keywords": []}
    all_annot_dict = {}
    for key in ["type", "start", "end", "duration", "score", "keywords"]:
        all_annot_dict[key] = [annot.get(key, "") for annot in all_annot]
    return all_annot_dict, seq_lengths


def calc_stats():
    annot_dir = os.path.join(BEAT_DIR, f"beat_English_15FPS_{len(JOI)}", "train", "annot")
    annot_files = os_sorted(glob.glob(pjoin(annot_dir, "*.txt")))
    all_annot, seq_lengths = collect_motion_labels(annot_files)
    hist = Counter(all_annot["type"])
    for k in BEAT_MOTION_TYPES:
        print(f"{k}: {hist[k]}")
        # 00_nogesture: 47
        # 01_beat_align: 14608, about 51%
        # 02_deictic_l: 413
        # 03_deictic_m: 954
        # 04_deictic_h: 1672
        # 05_iconic_l: 892
        # 06_iconic_m: 2259
        # 07_iconic_h: 2618
        # 08_metaphoric_l: 1192
        # 09_metaphoric_m: 2077
        # 10_metaphoric_h: 1589
        # habit: 188
        # need_cut: 306
    acc_stats = {}
    for label in BEAT_MOTION_TYPES:
        acc_stats[label] = {"counter": 0, "durations": []}
    total_length_labeled = 0
    for label, time in zip(all_annot["type"], all_annot["duration"]):
        acc_stats[label]["counter"] += 1
        acc_stats[label]["durations"].append(time)
        total_length_labeled += time
    for label in acc_stats:
        tmp = acc_stats[label]["durations"]
        n = len(tmp)
        sum = np.sum(tmp)
        avg = np.mean(tmp)
        std = np.std(tmp)
        print(f"{label}:\t{n} seqs; {sum:.3f} secs in total; mean {avg:.3f} secs; std {std:.3f} secs")

    pdb.set_trace()


if __name__ == "__main__":
    # inspect_joint_names(JOI)
    # visual_inspect()
    # pdb.set_trace()
    '''
    train_test = "test"
    # step 1: prepare train/val/test/special datasets (NOTE: special sequences in val/test are saved into special)
    # ********* ********* *********
    task_prep_train_val_test_data(
        BEAT_DIR,
        train_test=train_test,
        target_fps=TARGET_FPS,
        save_full_bvh=False,
        save_upper_bvh=False,
        overwrite=True,
    )
    '''
    
    # h5_dir = pjoin(BEAT_DIR, f"beat_English_{TARGET_FPS}FPS_{len(JOI)}")
    # # task_calc_per_speaker_stats(h5_dir, train_test)
    # save_fn = pjoin(h5_dir, f"{train_test}_stats.h5")
    # task_calc_dataset_stats(pjoin(h5_dir, train_test), train_test, save_fn)

    # save_fn = pjoin(h5_dir, "speaker_stats.h5")
    # speaker_stats = task_cmp_per_speaker_stats(h5_dir, save_fn)

    # print("Finished.")
    #pdb.set_trace()

    
    h5_fn = r"/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_English_15FPS_75/test/data/2_scott_0_103_103.h5"
    data = load_h5_dataset(h5_fn)
    print(list(data.keys()))
    print(data['joint_names'].shape)
    print("joint_names:",data['joint_names'])
    print(data['global_pos'].shape)
    print(data['rot_angles'].shape)
    print(data['rot_mats'].shape)
    print(data['wave16k'].shape)
    #print(data['euler_orders'])
    print(data['euler_orders'].shape)
    #print(data['rot_angles'])


    write("output_audio.wav", 16000, data['wave16k'])

    eulars=np.array(data['rot_angles'])
    rotmats=np.array(data['rot_mats'])
    euler_orders=np.array(data['euler_orders'])
    joint_names=np.array(data['joint_names'])
    print("rotmats shape:",rotmats.shape)
    np.savez("pred_rotmat_new1.npz",joint_names=joint_names,euler_orders=euler_orders,eulars=eulars,rotmats=rotmats)

    
    
