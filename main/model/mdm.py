import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention import LocalAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel



class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 96)
        self.fc2 = nn.Linear(96, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=128, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        self.audio_feat = audio_feat
        if audio_feat == 'wav encoder':
            self.audio_feat_dim = 32
        elif audio_feat == 'mfcc':
            self.audio_feat_dim = 13
        elif self.audio_feat == 'wavlm':
            print('USE WAVLM')
            self.audio_feat_dim = 64        # Linear 1024 -> 64
            self.WavEncoder_upper = WavEncoder()
            self.WavEncoder_hands = WavEncoder()
            self.WavEncoder_lower = WavEncoder()
            self.WavEncoder_exp = WavEncoder()

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)


        self.emb_trans_dec = emb_trans_dec

        self.latent_z_upper_linear = MyNetwork()
        self.latent_z_lower_linear = MyNetwork()
        self.latent_z_hands_linear = MyNetwork()
        self.latent_z_exp_linear = MyNetwork()

        self.cond_mode = cond_mode
        self.num_head = 8

        # 13*6    30*6     9*6+3     100
        if 'style2' not in self.cond_mode:
            self.input_process = InputProcess(self.data_rep, 433 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)
            self.input_process_upper = InputProcess(self.data_rep, 13*6 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)
            self.input_process_hands = InputProcess(self.data_rep, 30*6 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)
            self.input_process_lower = InputProcess(self.data_rep, 9*6+3 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)
            self.input_process_exp = InputProcess(self.data_rep, 100 + 3*6 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)


        if self.arch == 'mytrans_enc':
            print("MY TRANS_ENC init")
            from mytransformer import TransformerEncoderLayer, TransformerEncoder

            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(1536, self.latent_dim)

            seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_upper = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_upper = nn.TransformerEncoder(seqTransEncoderLayer_upper,
                                                         num_layers=self.num_layers)
            
            seqTransEncoderLayer_hands = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_hands = nn.TransformerEncoder(seqTransEncoderLayer_hands,
                                                         num_layers=self.num_layers)

            seqTransEncoderLayer_lower = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_lower = nn.TransformerEncoder(seqTransEncoderLayer_lower,
                                                         num_layers=self.num_layers)

            seqTransEncoderLayer_exp = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_exp = nn.TransformerEncoder(seqTransEncoderLayer_exp,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=False)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        if 'style1' in self.cond_mode:
            print('EMBED STYLE BEGIN TOKEN')
            if self.n_seed != 0:
                self.style_dim = 64
                self.embed_style_upper = nn.Linear(24, self.style_dim)
                self.embed_text_upper = nn.Linear((13*6) * n_seed, self.latent_dim - self.style_dim)
                self.embed_style_hands = nn.Linear(24, self.style_dim)
                self.embed_text_hands = nn.Linear((30*6) * n_seed, self.latent_dim - self.style_dim)
                self.embed_style_lower = nn.Linear(24, self.style_dim)
                self.embed_text_lower = nn.Linear((9*6+3) * n_seed, self.latent_dim - self.style_dim)
                self.embed_style_exp = nn.Linear(24, self.style_dim)
                self.embed_text_exp = nn.Linear((100+6*3) * n_seed, self.latent_dim - self.style_dim)


        self.output_process_upper = OutputProcess(self.data_rep, 13*6, self.latent_dim, 13*6,
                                            self.nfeats)
        self.output_process_hands = OutputProcess(self.data_rep, 30*6, self.latent_dim, 30*6,
                                            self.nfeats)
        self.output_process_lower = OutputProcess(self.data_rep, 9*6+3, self.latent_dim, 9*6+3,
                                            self.nfeats)
        self.output_process_exp = OutputProcess(self.data_rep, 100+3*6, self.latent_dim, 100+3*6,
                                            self.nfeats)

        if 'cross_local_attention' in self.cond_mode:
            self.rel_pos_upper = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.input_process = InputProcess(self.data_rep, 433 + self.gru_emb_dim, self.latent_dim//2)
            self.input_process_upper = InputProcess(self.data_rep, 13*6 + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention_upper = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2_upper = nn.Linear(self.latent_dim * 4 + 64 + self.audio_feat_dim, self.latent_dim)


            self.rel_pos_hands = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.input_process_hands = InputProcess(self.data_rep, 30*6 + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention_hands = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2_hands = nn.Linear(self.latent_dim * 4 + 64 + self.audio_feat_dim, self.latent_dim)

            self.rel_pos_lower = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.input_process_lower = InputProcess(self.data_rep, 9*6+3 + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention_lower = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2_lower = nn.Linear(self.latent_dim * 4 + 64 + self.audio_feat_dim, self.latent_dim)


            self.rel_pos_exp = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            # self.input_process_exp = InputProcess(self.data_rep, 100 + self.gru_emb_dim, self.latent_dim)
            self.input_process_exp = InputProcess(self.data_rep, 100+3*6 + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention_exp = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2_exp = nn.Linear(self.latent_dim * 4 + 64 + self.audio_feat_dim, self.latent_dim)

        if 'cross_local_attention2' in self.cond_mode:
            print('Cross Local Attention2')
            self.selfAttention = LinearTemporalCrossAttention(seq_len=0, latent_dim=128, text_latent_dim=256, num_head=8, dropout=0.1, time_embed_dim=0)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps,latent_z_upper,latent_z_lower,latent_z_hands,latent_z_exp, y=None,uncond_info=False):
        lz_upper = self.latent_z_upper_linear(latent_z_upper).unsqueeze(0)#bs,256
        lz_lower = self.latent_z_lower_linear(latent_z_lower).unsqueeze(0)#bs,256
        lz_hands = self.latent_z_hands_linear(latent_z_hands).unsqueeze(0)#bs,256
        lz_exp = self.latent_z_exp_linear(latent_z_exp).unsqueeze(0)#bs,256

        text=y['text']
        text_latent_upper = y['text_latent_upper']
        text_latent_hands = y['text_latent_hands']
        text_latent_lower = y['text_latent_lower']
        text_latent_exp = y['text_latent_exp']

        bs, njoints, nfeats, nframes = x.shape      # 64, 251, 1, 196

        if y['use_arms'] and y['use_lower']:
            x_upper = y['arms']
            x_hands = y['hands']*0.0
            x_lower = y['lower']
            x_exp = y['exp']
            x = torch.cat([x_upper,x_hands,x_lower,x_exp],axis=1)
        else:
            x_upper = x[:,:13*6,:,:]
            x_hands = x[:,13*6:13*6+30*6,:,:]
            x_lower = x[:,43*6:43*6+9*6+3,:,:]
            x_exp = x[:,3+52*6:,:,:]


        emb_t = self.embed_timestep(timesteps)  # [1, bs, d], (1, 2, 256)

        force_mask=False
        
        if 'style1' in self.cond_mode:
            embed_style_upper = self.mask_cond(self.embed_style_upper(y['style']), force_mask=force_mask)       # (bs, 64)
            if self.n_seed != 0:
                embed_text_upper = self.embed_text_upper(self.mask_cond(y['seed'][:,:13*6,:,:].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                emb_1_upper = torch.cat((embed_style_upper, embed_text_upper), dim=1)
            style_hands = torch.as_tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float().to('cuda')
            embed_style_hands = self.mask_cond(self.embed_style_hands(style_hands), force_mask=force_mask)       # (bs, 64)
            if self.n_seed != 0:
                embed_text_hands = self.embed_text_hands(self.mask_cond(y['seed'][:,13*6:13*6+30*6,:,:].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                emb_1_hands = torch.cat((embed_style_hands, embed_text_hands), dim=1)

            embed_style_lower = self.mask_cond(self.embed_style_lower(y['style']), force_mask=force_mask)       # (bs, 64)
            if self.n_seed != 0:
                embed_text_lower = self.embed_text_lower(self.mask_cond(y['seed'][:,43*6:43*6+9*6+3,:,:].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                emb_1_lower = torch.cat((embed_style_lower, embed_text_lower), dim=1)

            embed_style_exp = self.mask_cond(self.embed_style_exp(y['style']), force_mask=force_mask)       # (bs, 64)
            if self.n_seed != 0:
                embed_text_exp = self.embed_text_exp(self.mask_cond(y['seed'][:,3+52*6:,:,:].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                emb_1_exp = torch.cat((embed_style_exp, embed_text_exp), dim=1)


        if self.audio_feat == 'wavlm':
            enc_text_upper = self.WavEncoder_upper(y['audio']).permute(1, 0, 2)
            enc_text_hands = self.WavEncoder_hands(y['audio']).permute(1, 0, 2)
            enc_text_lower = self.WavEncoder_lower(y['audio']).permute(1, 0, 2)
            enc_text_exp = self.WavEncoder_exp(y['audio']).permute(1, 0, 2)


        if 'cross_local_attention' in self.cond_mode:
            if 'cross_local_attention3' in self.cond_mode:
                x_ = x.reshape(bs, 433, 1, nframes) 
                x_upper = x_upper.reshape(bs, 13*6, 1, nframes)  # [2, 135, 1, 240]
                x_hands = x_hands.reshape(bs, 30*6, 1, nframes)  # [2, 135, 1, 240]
                x_lower = x_lower.reshape(bs, 9*6+3, 1, nframes)  # [2, 135, 1, 240]
                x_exp = x_exp.reshape(bs, 100+3*6, 1, nframes)  # [2, 135, 1, 240]
                # self-attention
                x__upper = self.input_process_upper(x_upper)  # [2, 135, 1, 240] -> [240, 2, 256]
                x__ = self.input_process(x_)
                x__hands = self.input_process_hands(x_hands)
                x__lower = self.input_process_lower(x_lower)
                x__exp = self.input_process_exp(x_exp)

                # local-cross-attention
                packed_shape = [torch.Size([bs, self.num_head])]
                xseq_upper = torch.cat((x__upper, enc_text_upper, x__), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32)
                xseq_hands = torch.cat((x__hands, enc_text_hands, x__), axis=2)
                xseq_lower = torch.cat((x__lower, enc_text_lower, x__), axis=2)
                xseq_exp = torch.cat((x__exp, enc_text_exp, x__), axis=2)
                # all frames
                embed_style_2_upper = torch.cat([(emb_1_upper + emb_t),lz_upper],axis=-1).repeat(nframes, 1, 1)  # (bs, 64) -> (len, bs, 64)
                embed_style_2_hands = torch.cat([(emb_1_hands + emb_t),lz_hands],axis=-1).repeat(nframes, 1, 1)
                embed_style_2_lower = torch.cat([(emb_1_lower + emb_t),lz_lower],axis=-1).repeat(nframes, 1, 1)
                embed_style_2_exp = torch.cat([(emb_1_exp + emb_t),lz_exp],axis=-1).repeat(nframes, 1, 1)

                text_latent_upper_local=text_latent_upper.squeeze(1).repeat(nframes, 1, 1)
                text_latent_hands_local=text_latent_hands.squeeze(1).repeat(nframes, 1, 1)
                text_latent_lower_local=text_latent_lower.squeeze(1).repeat(nframes, 1, 1)
                text_latent_exp_local=text_latent_exp.squeeze(1).repeat(nframes, 1, 1)
                embed_style_2_upper=torch.cat((embed_style_2_upper,text_latent_upper_local),axis=2)
                embed_style_2_hands=torch.cat((embed_style_2_hands,text_latent_hands_local),axis=2)
                embed_style_2_lower=torch.cat((embed_style_2_lower,text_latent_lower_local),axis=2)
                embed_style_2_exp=torch.cat((embed_style_2_exp,text_latent_exp_local),axis=2)
                xseq_upper = torch.cat((embed_style_2_upper, xseq_upper), axis=2)  # (seq, bs, dim)
                xseq_upper = self.input_process2_upper(xseq_upper)
                xseq_upper = xseq_upper.permute(1, 0, 2)  # (bs, len, dim)
                xseq_upper = xseq_upper.view(bs, nframes, self.num_head, -1)
                xseq_upper = xseq_upper.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_upper = xseq_upper.reshape(bs * self.num_head, nframes, -1)
                pos_emb_upper = self.rel_pos_upper(xseq_upper)  # (89, 32)
                xseq_upper, _ = apply_rotary_pos_emb(xseq_upper, xseq_upper, pos_emb_upper)
                xseq_upper = self.cross_local_attention_upper(xseq_upper, xseq_upper, xseq_upper, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (2, 8, 2048, 64)
                xseq_upper = xseq_upper.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                xseq_upper = xseq_upper.reshape(bs, nframes, -1)
                xseq_upper = xseq_upper.permute(1, 0, 2)
                
                xseq_upper = torch.cat((emb_1_upper + emb_t,lz_upper,text_latent_upper.unsqueeze(0), xseq_upper), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq_upper = xseq_upper.permute(1, 0, 2)  # (bs, len, dim)
                xseq_upper = xseq_upper.view(bs, nframes + 3, self.num_head, -1)
                xseq_upper = xseq_upper.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_upper = xseq_upper.reshape(bs * self.num_head, nframes + 3, -1)
                pos_emb_upper = self.rel_pos_upper(xseq_upper)  # (89, 32)
                xseq_upper, _ = apply_rotary_pos_emb(xseq_upper, xseq_upper, pos_emb_upper)
                xseq_rpe_upper = xseq_upper.reshape(bs, self.num_head, nframes + 3, -1)
                xseq_upper = xseq_rpe_upper.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                xseq_upper = xseq_upper.view(bs, nframes + 3, -1)
                xseq_upper = xseq_upper.permute(1, 0, 2)


                #hands
                xseq_hands = torch.cat((embed_style_2_hands, xseq_hands), axis=2)  # (seq, bs, dim)
                xseq_hands = self.input_process2_hands(xseq_hands)
                xseq_hands = xseq_hands.permute(1, 0, 2)  # (bs, len, dim)
                xseq_hands = xseq_hands.view(bs, nframes, self.num_head, -1)
                xseq_hands = xseq_hands.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_hands = xseq_hands.reshape(bs * self.num_head, nframes, -1)
                pos_emb_hands = self.rel_pos_hands(xseq_hands)  # (89, 32)
                xseq_hands, _ = apply_rotary_pos_emb(xseq_hands, xseq_hands, pos_emb_hands)
                xseq_hands = self.cross_local_attention_hands(xseq_hands, xseq_hands, xseq_hands, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (2, 8, 2048, 64)
                xseq_hands = xseq_hands.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                xseq_hands = xseq_hands.reshape(bs, nframes, -1)
                xseq_hands = xseq_hands.permute(1, 0, 2)

                xseq_hands = torch.cat((emb_1_hands + emb_t,lz_hands,text_latent_hands.unsqueeze(0), xseq_hands), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq_hands = xseq_hands.permute(1, 0, 2)  # (bs, len, dim)
                xseq_hands = xseq_hands.view(bs, nframes + 3, self.num_head, -1)
                xseq_hands = xseq_hands.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_hands = xseq_hands.reshape(bs * self.num_head, nframes + 3, -1)
                pos_emb_hands = self.rel_pos_hands(xseq_hands)  # (89, 32)
                xseq_hands, _ = apply_rotary_pos_emb(xseq_hands, xseq_hands, pos_emb_hands)
                xseq_rpe_hands = xseq_hands.reshape(bs, self.num_head, nframes + 3, -1)
                xseq_hands = xseq_rpe_hands.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                xseq_hands = xseq_hands.view(bs, nframes + 3, -1)
                xseq_hands = xseq_hands.permute(1, 0, 2)

                #lower
                xseq_lower = torch.cat((embed_style_2_lower, xseq_lower), axis=2)  # (seq, bs, dim)
                xseq_lower = self.input_process2_lower(xseq_lower)
                xseq_lower = xseq_lower.permute(1, 0, 2)  # (bs, len, dim)
                xseq_lower = xseq_lower.view(bs, nframes, self.num_head, -1)
                xseq_lower = xseq_lower.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_lower = xseq_lower.reshape(bs * self.num_head, nframes, -1)
                pos_emb_lower = self.rel_pos_lower(xseq_lower)  # (89, 32)
                xseq_lower, _ = apply_rotary_pos_emb(xseq_lower, xseq_lower, pos_emb_lower)
                xseq_lower = self.cross_local_attention_lower(xseq_lower, xseq_lower, xseq_lower, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (2, 8, 2048, 64)
                xseq_lower = xseq_lower.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                xseq_lower = xseq_lower.reshape(bs, nframes, -1)
                xseq_lower = xseq_lower.permute(1, 0, 2)

                xseq_lower = torch.cat((emb_1_lower + emb_t,lz_lower,text_latent_lower.unsqueeze(0), xseq_lower), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq_lower = xseq_lower.permute(1, 0, 2)  # (bs, len, dim)
                xseq_lower = xseq_lower.view(bs, nframes + 3, self.num_head, -1)
                xseq_lower = xseq_lower.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_lower = xseq_lower.reshape(bs * self.num_head, nframes + 3, -1)
                pos_emb_lower = self.rel_pos_lower(xseq_lower)  # (89, 32)
                xseq_lower, _ = apply_rotary_pos_emb(xseq_lower, xseq_lower, pos_emb_lower)
                xseq_rpe_lower = xseq_lower.reshape(bs, self.num_head, nframes + 3, -1)
                xseq_lower = xseq_rpe_lower.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                xseq_lower = xseq_lower.view(bs, nframes + 3, -1)
                xseq_lower = xseq_lower.permute(1, 0, 2)

                #exp
                xseq_exp = torch.cat((embed_style_2_exp, xseq_exp), axis=2)  # (seq, bs, dim)
                xseq_exp = self.input_process2_exp(xseq_exp)
                xseq_exp = xseq_exp.permute(1, 0, 2)  # (bs, len, dim)
                xseq_exp = xseq_exp.view(bs, nframes, self.num_head, -1)
                xseq_exp = xseq_exp.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_exp = xseq_exp.reshape(bs * self.num_head, nframes, -1)
                pos_emb_exp = self.rel_pos_exp(xseq_exp)  # (89, 32)
                xseq_exp, _ = apply_rotary_pos_emb(xseq_exp, xseq_exp, pos_emb_exp)
                xseq_exp = self.cross_local_attention_exp(xseq_exp, xseq_exp, xseq_exp, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (2, 8, 2048, 64)
                xseq_exp = xseq_exp.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                xseq_exp = xseq_exp.reshape(bs, nframes, -1)
                xseq_exp = xseq_exp.permute(1, 0, 2)

                xseq_exp = torch.cat((emb_1_exp + emb_t,lz_exp,text_latent_exp.unsqueeze(0), xseq_exp), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq_exp = xseq_exp.permute(1, 0, 2)  # (bs, len, dim)
                xseq_exp = xseq_exp.view(bs, nframes + 3, self.num_head, -1)
                xseq_exp = xseq_exp.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq_exp = xseq_exp.reshape(bs * self.num_head, nframes + 3, -1)
                pos_emb_exp = self.rel_pos_exp(xseq_exp)  # (89, 32)
                xseq_exp, _ = apply_rotary_pos_emb(xseq_exp, xseq_exp, pos_emb_exp)
                xseq_rpe_exp = xseq_exp.reshape(bs, self.num_head, nframes + 3, -1)
                xseq_exp = xseq_rpe_exp.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                xseq_exp = xseq_exp.view(bs, nframes + 3, -1)
                xseq_exp = xseq_exp.permute(1, 0, 2)


                #print("xseq_upper shape:",xseq_upper.shape)#73 300 256
                output_upper = self.seqTransEncoder_upper(xseq_upper)[3:]
                output_hands = self.seqTransEncoder_hands(xseq_hands)[3:]
                output_lower = self.seqTransEncoder_lower(xseq_lower)[3:]
                output_exp = self.seqTransEncoder_exp(xseq_exp)[3:]

        output_upper = self.output_process_upper(output_upper)  # [bs, njoints, nfeats, nframes]
        output_hands = self.output_process_hands(output_hands)
        output_lower = self.output_process_lower(output_lower)
        output_exp = self.output_process_exp(output_exp)


        output=torch.cat([output_upper,output_hands,output_lower,output_exp],axis=1)

        return output.to("cuda"),output_upper.to("cuda"),output_hands.to("cuda"),output_lower.to("cuda"),output_exp.to("cuda")


    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_map = nn.Linear(1024, 64)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep


if __name__ == '__main__':
    '''
    cd ./main/model
    python mdm.py
    '''
    n_frames = 240

    n_seed = 8

    model = MDM(modeltype='', njoints=1140, nfeats=1, cond_mode = 'cross_local_attention5_style1', action_emb='tensor', audio_feat='mfcc',
                arch='mytrans_enc', latent_dim=256, n_seed=n_seed, cond_mask_prob=0.1)

    x = torch.randn(2, 1140, 1, 88)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(2, 88, 13).permute(1, 0, 2)       # [n_seed:, ...]
    model_kwargs_['y']['style'] = torch.randn(2, 6)
    model_kwargs_['y']['mask_local'] = torch.ones(2, 88).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]
    y = model(x, t, model_kwargs_['y'])
    print(y.shape)
