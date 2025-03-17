# Efficient-Audio-Gesture

This is the official repository of the two papers. 

(👀 DIDiffGes elaborates on the process and inference of semi-implicit accelerated diffusion. This part plays a crucial role in the efficient generation of HoloGest.)

**🔥(AAAI 2025) DIDiffGes: Decoupled Semi-Implicit Diffusion Models for Real-time Gesture Generation from Speech**
> *The 39th Annual AAAI Conference on Artificial Intelligence (AAAI), 2025*

**[[paper is here!](https://hologest.github.io/)]**

**🔥(3DV 2025) HoleGest: Decoupled Diffusion and Motion Priors for Generating Holisticly Expressive Co-speech Gestures**
> *International Conference on 3D Vision 2025 (3DV), 2025*

**[[Project Page](https://hologest.github.io/)]** **[[Paper](https://hologest.github.io/)]** 

# Method
![Image](https://github.com/user-attachments/assets/4472d621-1fb1-4c6b-a155-2ccfa5a8532c)

We used an avatar to conduct an audio narration of our method, vividly elaborating on the details of our method to everyone.

https://github.com/user-attachments/assets/204a0806-5623-4fbe-a1e7-6f150279ea1a

# News :triangular_flag_on_post:
- [2024/12/15] **DIDiffGes got accepted by AAAI 2025!** 🎉
- [2024/11/10] **HoloGest got accepted by 3DV 2025!** 🎉


## 1. Getting started

This code was tested on `NVIDIA GeForce RTX 3070 Ti` and requires:

* conda3 or miniconda3

```
cd ./main/
bash pip_install.sh
```

## 2. Quick Start

```
bash demo.sh
```

## Citation
```
@inproceedings{yu2023acr,
  title = {ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction},
  author = {Yu, Zhengdi and Huang, Shaoli and Chen, Fang and Breckon, Toby P. and Wang, Jue},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023}
  }
```

## Acknowledgement
The pytorch implementation of HoloGest is based on [ExpGest](https://github.com/cyk990422/ExpGest). We use some parts of the knowledgement from [SiDDMs](https://arxiv.org/abs/2306.12511) and some part of code from [DIDiffGes]. We thank all the authors for their impressive works!

## Contact
For technical questions, please contact cyk19990422@gmail.com

For commercial licensing, please contact shaolihuang@tencent.com
