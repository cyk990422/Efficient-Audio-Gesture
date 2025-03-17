import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model', '../../libs', '../../libs/process']]
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

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor


from transformers import AutoTokenizer, AutoModel

from torch.nn import functional as F
import torch
import torch.nn as nn

#Semantical embedding
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="../model/te/1dbc166cf8765166998eff31ade2eb64c8a40076", output_dim=128):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)



        seqTransEncoderLayer_exp = nn.TransformerEncoderLayer(d_model=128,
                                                              nhead=4,
                                                              dim_feedforward=1024,
                                                              dropout=0.1,
                                                              activation="gelu")

        self.seqTransEncoder_exp = nn.TransformerEncoder(seqTransEncoderLayer_exp,
                                                        num_layers=8)

        self.embed_style = nn.Linear(24, 128)

    def forward(self, texts, style):
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_output.last_hidden_state

        cls_token = hidden_states[:, 0, :]

        features = self.linear(cls_token)

        features_res = features.reshape(1,features.shape[0],-1)
        style = style.to("cuda")
        style_emb = self.embed_style(style).reshape(1,features.shape[0],-1)

        pred_input = torch.cat([style_emb,features_res],axis=0)
        pred_output = self.seqTransEncoder_exp(pred_input)[1:]
        pred_output_res = pred_output.reshape(features.shape[0],-1)


        return features, pred_output_res

class SemGes_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.Textencoder_upper=TextEncoder()
        self.Textencoder_hands=TextEncoder()
        self.Textencoder_lower=TextEncoder()
        self.Textencoder_exp=TextEncoder()

    def forward(self,text,style):
        # Text Encoder to obtain the same shape latent code
        _, text_latent_upper=self.Textencoder_upper(text,style)
        style_hands = torch.as_tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float().to('cuda')
        _, text_latent_hands=self.Textencoder_hands(text,style_hands)
        _, text_latent_lower=self.Textencoder_lower(text,style)
        _, text_latent_exp=self.Textencoder_exp(text,style)

        return text_latent_upper, text_latent_hands, text_latent_lower, text_latent_exp


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

    def file_to_text(self, audio_input):
        samplerate = 16000

        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


asr = Wave2Vec2Inference("./asr/54074b1c16f4de6a5ad59affb4caa8f2ea03a119")


def batch_rot2aa(Rs):
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


def rotation_matrix_to_angle_axis(rotation_matrix):
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
                glob_rot=True, cond_mode = 'cross_local_attention3_style1', clip_version = 'ViT-B/32', action_emb = 'tensor', audio_feat='wavlm',
                arch='trans_enc', latent_dim=128, n_seed=12)        # trans_enc, trans_dec, gru, mytrans_enc
    diffusion = create_gaussian_diffusion()
    return model, diffusion





def inference(args, wavlm_model, audio, sample_fn, model,mydevice, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=0, n_seed=12, style=None, seed=123456):
    batch_size = 1 
    torch.manual_seed(seed)

    if n_frames == 0:
        n_frames = audio.shape[0] * 30 // 16000
    if minibatch:
        stride_poses = args.n_poses - n_seed
        if n_frames < stride_poses:
            num_subdivision = 1
        else:
            num_subdivision = math.ceil(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses
            print(
                '{}, {}, {}'.format(num_subdivision, stride_poses, n_frames))
    audio = audio[:n_frames * int(16000 / 30)]
    print("num_subdivision*32000:",num_subdivision*32000,"      ",audio.shape[0])
    audio_padding = np.zeros((num_subdivision*32000 - audio.shape[0]))
    print("shape:",audio.shape,"    ",n_frames,"     ",audio_padding.shape)
    audio = np.concatenate((audio,audio_padding),axis = 0)

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    semges=SemGes_Dis()
    checkpoint=torch.load("../model/vae_checkpoint_1000.bin")
    
    semges.load_state_dict(checkpoint['state_dict'], strict=False)
    semges.eval()
    semges.cuda()


    sample_list_demo=[]

    if minibatch:
        audio_reshape = torch.from_numpy(audio).to(torch.float32).reshape(num_subdivision, int(stride_poses * 16000 / 30)).to(mydevice).transpose(0, 1)       # mfcc[:, :-2]
        shape_ = (1, model.njoints, model.nfeats, args.n_poses)
        out_list = []
        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]

            if i == 0:
                if n_seed != 0:
                    pad_zeros = torch.zeros([int(n_seed * 16000 / 30), 1]).to(mydevice)        # wavlm dims are 1024
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = torch.zeros([1, 433, 1, n_seed]).to(mydevice)
                    save_audio_clip=model_kwargs_['y']['audio'].clone()
                    save_audio_clip=save_audio_clip[:,0].detach().cpu().numpy()
                    save_audio_clip=save_audio_clip.astype(np.float32)
                    text = asr.file_to_text(save_audio_clip)
                    sample_text=text[0]
                    model_kwargs_['y']['text']=sample_text
                    model_kwargs_['y']['use_hints'] = False
                    with torch.no_grad():
                        text_latent_upper, text_latent_hands, text_latent_lower, text_latent_exp=semges(sample_text,model_kwargs_['y']['style'])

                    model_kwargs_['y']['text_latent_upper']=text_latent_upper
                    model_kwargs_['y']['text_latent_hands']=text_latent_hands
                    model_kwargs_['y']['text_latent_lower']=text_latent_lower
                    model_kwargs_['y']['text_latent_exp']=text_latent_exp
            else:
                if n_seed != 0:
                    pad_audio = audio_reshape[-int(n_seed * 16000 / 30):, i - 1:i]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = out_list[-1][..., -n_seed:].to(mydevice)

                    save_audio_clip=model_kwargs_['y']['audio'].clone()
                    save_audio_clip=save_audio_clip[:,0].detach().cpu().numpy()
                    save_audio_clip=save_audio_clip.astype(np.float32)
                    text = asr.file_to_text(save_audio_clip)
                    sample_text=text[0]
                    model_kwargs_['y']['text']=sample_text
                    if i == 2:
                        model_kwargs_['y']['use_hints'] = False
                    else:
                        model_kwargs_['y']['use_hints'] = False
                    with torch.no_grad():
                        text_latent_upper, text_latent_hands, text_latent_lower, text_latent_exp=semges(sample_text,model_kwargs_['y']['style'])
                    model_kwargs_['y']['text_latent_upper']=text_latent_upper
                    model_kwargs_['y']['text_latent_hands']=text_latent_hands
                    model_kwargs_['y']['text_latent_lower']=text_latent_lower
                    model_kwargs_['y']['text_latent_exp']=text_latent_exp



            model_kwargs_['y']['audio'] = wav2wavlm(wavlm_model, model_kwargs_['y']['audio'].transpose(0, 1), mydevice)

            model_kwargs_['y']['use_arms'] = False
            model_kwargs_['y']['use_lower'] = False

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
            )
            sample_list_demo.append(sample)
            

            
            # smoothing motion transition
            if len(out_list) > 0 and n_seed != 0:
                last_poses = out_list[-1][..., -n_seed:]        # # (1, model.njoints, 1, n_seed)
                out_list[-1] = out_list[-1][..., :-n_seed]  # delete last 4 frames
                if smoothing:
                    # Extract predictions
                    last_poses_root_pos = last_poses[:, 53*6:53*6+3]        # (1, 3, 1, 8)
                    next_poses_root_pos = sample[:, 53*6:53*6+3]        # (1, 3, 1, 88)
                    root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                    predict_pos = next_poses_root_pos[..., 0]
                    delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                    sample[:, 53*6:53*6+3] = sample[:, 53*6:53*6+3] - delta_pos

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            
            out_list.append(sample)
        sample_list_demo = [i.detach().data.cpu().numpy() for i in sample_list_demo]
        # np.save("sample_list_demo.npy",np.array(sample_list_demo))
        if n_seed != 0:
            out_list[-1] = out_list[-1][..., :-n_seed]
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
            sampled_seq = sampled_seq[:, n_seed:]

        else:
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, 72, model.njoints)

    data_mean_ = np.load("./mean.npz")['mean'].squeeze()
    data_std_ = np.load("./std.npz")['std'].squeeze()

    data_mean = np.array(data_mean_).squeeze()
    data_std = np.array(data_std_).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    
    upper_index=[3,6,9,12,13,14,15,16,17,18,19,20,21]
    lower_index=[0,1,2,4,5,7,8,10,11]

    body_index=[13,14,15,0,16,17,1,18,19,2,20,21,3,4,5,6,7,8,9,10,11,12]

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

    mean_jaw = data_mean_6d[22:25,:].reshape(-1)
    std_jaw = std_6d[22:25,:].reshape(-1)

    mean_exp = data_mean[330:430]
    std_exp = std[330:430]
    mean_exp = np.concatenate((mean_exp,mean_jaw),axis=-1)
    std_exp = np.concatenate((std_exp,std_jaw),axis=-1)

    data_mean = np.concatenate((mean_upper,mean_hands,mean_lower_trans,mean_exp),axis=0)
    std = np.concatenate((std_upper,std_hands,std_lower_trans,std_exp),axis=0)

    out_poses = np.multiply(sampled_seq[0], std) + data_mean


    data_rot6d_flat = out_poses.copy()
    n_poses = data_rot6d_flat.shape[0]
    out_poses = np.zeros((n_poses, data_rot6d_flat.shape[1]))
    for i in range(out_poses.shape[1]):
        out_poses[:, i] = savgol_filter(data_rot6d_flat[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal


    #(56, 421)
    out_poses_rot6d_upper = out_poses[:,:13*6].reshape(-1,13,6)
    out_poses_rot6d_hands = out_poses[:,13*6:43*6].reshape(-1,30,6)
    out_poses_rot6d_lower = out_poses[:,43*6:52*6].reshape(-1,9,6)
    out_poses_trans = out_poses[:,52*6:52*6+3].reshape(-1,3)
    out_poses_exp = out_poses[:,52*6+3:52*6+103].reshape(-1,100)
    out_poses_face = out_poses[:,52*6+103:52*6+103+18].reshape(-1,3,6)
    
    out_poses_rot6d_body = np.concatenate((out_poses_rot6d_upper,out_poses_rot6d_lower),axis=1)
    out_poses_rot6d_body = out_poses_rot6d_body[:,body_index,:]
    

    out_poses_rot6d = torch.tensor(np.concatenate((out_poses_rot6d_body,out_poses_face,out_poses_rot6d_hands),axis=1).reshape(-1,6)).cuda()
    out_poses_rotmat = rot6d_to_rotmat(out_poses_rot6d)#-1,3,3
    out_poses_aa = batch_rot2aa(out_poses_rotmat)#-1,3
    out_poses_aa = out_poses_aa.detach().cpu().numpy().reshape(-1,55,3)

    np.savez("beat2_our.npz",poses=out_poses_aa,trans=out_poses_trans,betas=np.zeros(10),expression=out_poses_exp,gender='male',mocap_framerate=30,)

    return out_poses_aa, out_poses_trans, out_poses_exp



def main(args, save_dir, model_path,speaker_style, mydevice, audio_path=None, mfcc_path=None, audiowavlm_path=None, max_len=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if audiowavlm_path != None:
        mfcc, fs = librosa.load(audiowavlm_path, sr=16000)

    elif audio_path != None and mfcc_path == None:
        # normalize_audio
        audio_name = audio_path.split('/')[-1]
        print('normalize audio: ' + audio_name)
        normalize_wav_path = os.path.join(save_dir, 'normalize_' + audio_name)
        cmd = ['ffmpeg-normalize', audio_path, '-o', normalize_wav_path, '-ar', '16000']
        subprocess.call(cmd)

        # MFCC, https://github.com/supasorn/synthesizing_obama_network_training
        print('extract MFCC...')
        obj = MFCC(frate=20)
        wav, fs = librosa.load(normalize_wav_path, sr=16000)
        mfcc = obj.sig2s2mfc_energy(wav, None)
        print(mfcc[:, :-2].shape)  # -1 -> -2      # (502, 13)
        np.savez_compressed(os.path.join(save_dir, audio_name[:-4] + '.npz'), mfcc=mfcc[:, :-2])

    elif mfcc_path != None and audio_path == None:
        mfcc = np.load(mfcc_path)['mfcc']

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()

    sample_fn = diffusion.p_sample_loop     # predict x_start

    style2onehot = {
        'wayne':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'scott':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'solomon':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lawrence':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'stewart':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'carla':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'sophie':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'miranda':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'kieks':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'nidal':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lu':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'carlos':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'jorge':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'itoi':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'daiki':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'li':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'ayana':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'kaita':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'hailing':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'kexin':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'goto':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'yingqing':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'tiffnay':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'katya':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }

    style = style2onehot[speaker_style]

    wavlm_model = wavlm_init(mydevice)
    inference(args, wavlm_model, mfcc, sample_fn, model,mydevice, n_frames=max_len, smoothing=True, SG_filter=True, minibatch=True, skip_timesteps=0, style=style, seed=-6266)      # style2onehot['Happy']


import random

def init_A2G_model(model_path):
    save_dir = 'sample_dir'

    parser = argparse.ArgumentParser(description='HoloGest')
    parser.add_argument('--config', default='./configs/HoloGest.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default=model_path)
    parser.add_argument('--audiowavlm_path', type=str, default='./test_audio.wav')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--speaker_style', type=str, default='lu')
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

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()
    return config, model, diffusion


if __name__ == '__main__':
    save_dir = 'sample_dir'

    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--model_path', type=str, default='./model000450000.pt')
    parser.add_argument('--audiowavlm_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--speaker_style', type=str, default='wayne')
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

    main(config, save_dir, config.model_path,config.speaker_style,mydevice, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len)

