""" create data samples """
import pdb

import lmdb
import math
import numpy as np
import pyarrow


import torch
import torch.nn.functional as F


import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

# Improvements: 
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_lm_if_possible:            
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits            

        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,  
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)

        return transcription, confidence   

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


asr = Wave2Vec2Inference("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/my_model1/huggingface/hub/models--facebook--wav2vec2-large-960h-lv60-self/snapshots/54074b1c16f4de6a5ad59affb4caa8f2ea03a119")

def wavlm_init(device=torch.device('cuda:1')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    # wavlm_model_path = '../../../My/process/WavLM-Base+.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

#from funasr_onnx import Paraformer

def wav2wavlm(model, wav_input_16khz, device=torch.device('cuda:1')):
    with torch.no_grad():
        wav_input_16khz = torch.from_numpy(wav_input_16khz).float()
        wav_input_16khz = wav_input_16khz.to(device).unsqueeze(0)
        rep = model.extract_features(wav_input_16khz)[0]
        rep = F.interpolate(rep.transpose(1, 2), size=72, align_corners=True, mode='linear').transpose(1, 2)
        return rep.squeeze().cpu().detach().data.cpu().numpy()


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, device):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 1024 * 20  # in TB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.model = wavlm_init(device)
        self.device = device

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                print("clip_idx:  ",str(clip_idx)+" / "+str(len(clips)))
                self._sample_from_clip(vid, clip, self.device)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])
        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()


    def _sample_from_clip(self, vid, clip, device):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        clip_styles_raw = clip['style_raw']
        clip_mfcc_raw = clip['mfcc_raw']
        

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_audio_list = []
        sample_text_list = []
        sample_codes_list = []
        sample_mfcc_list = []
        sample_wavlm_list = []

        MINLEN = min(len(clip_skeleton), int(len(clip_audio_raw) * 60 / 16000), len(clip_mfcc_raw))

        num_subdivision = math.floor(
            (MINLEN - self.n_poses)
            / self.subdivision_stride)  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            print("i /  num_subdivision:  ",str(i)+"  /   "+str(num_subdivision))
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            sample_mfcc = clip_mfcc_raw[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]
            
            save_audio_clip=sample_audio.copy()
            #print("save_audio_clip shape:",save_audio_clip.shape)
            save_audio_clip=save_audio_clip.astype(np.float32)
            sr = 16000
            audio_path = 'test_audio.wav'
            sf.write(audio_path, save_audio_clip, sr)
            
            text = asr.file_to_text(audio_path)
            sample_text=text[0].lower()
            '''
            model_dir = "/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/DiffuseStyleGesture/main/mydiffusion_zeggs/wav2vec2-live/paraformer-large"
            model = Paraformer(model_dir, batch_size=1, quantize=True)

            wav_path = [audio_path]
            sample_text = model(wav_path)[0]['preds'][0]
            '''
            #sample_text=''
            print("sample_text:   ",sample_text)
            
            
            sample_wavlm = wav2wavlm(self.model, sample_audio, device=device)

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_mfcc_list.append(sample_mfcc)
            sample_wavlm_list.append(sample_wavlm)
            sample_audio_list.append(sample_audio)
            sample_text_list.append(sample_text)
            sample_codes_list.append(clip_styles_raw)
            aux_info.append(motion_info)

        # if len(sample_skeletons_list) > 0:
        #     with self.dst_lmdb_env.begin(write=True) as txn:
        #         for poses, audio, codes, mfcc, wavlm, aux in zip(sample_skeletons_list,
        #                                             sample_audio_list, sample_codes_list, sample_mfcc_list, sample_wavlm_list, aux_info):
        #             poses = np.asarray(poses)
        #
        #             # save
        #             k = '{:010}'.format(self.n_out_samples).encode('ascii')
        #             v = [poses, audio, codes, mfcc, wavlm, aux]
        #             v = pyarrow.serialize(v).to_buffer()
        #             txn.put(k, v)
        #             self.n_out_samples += 1

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, codes, wavlm,text in zip(sample_skeletons_list, sample_codes_list, sample_wavlm_list,sample_text_list):
                    poses = np.asarray(poses)
                    print("poses shape:",poses.shape)
                    
                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [poses, codes, wavlm,text]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

