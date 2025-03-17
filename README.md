# Efficient-Audio-Gesture

This is the official repository of the two papers. 

(👀 DIDiffGes elaborates on the process and inference of semi-implicit accelerated diffusion. This part plays a crucial role in the efficient generation of HoloGest.)

**🔥(AAAI 2025) DIDiffGes: Decoupled Semi-Implicit Diffusion Models for Real-time Gesture Generation from Speech**
> *The 39th Annual AAAI Conference on Artificial Intelligence (AAAI), 2025*

**[[paper is here!](https://hologest.github.io/)]**

**🔥(3DV 2025) HoleGest: Decoupled Diffusion and Motion Priors for Generating Holisticly Expressive Co-speech Gestures**
> *International Conference on 3D Vision 2025 (3DV), 2025*

**[[Project Page](https://cyk990422.github.io/HoloGest.github.io/)]** **[[Paper](https://hologest.github.io/)]** 

# Method
![Image](https://github.com/user-attachments/assets/4472d621-1fb1-4c6b-a155-2ccfa5a8532c)

We used an avatar to conduct an audio narration of our method, vividly elaborating on the details of our method to everyone.

https://github.com/user-attachments/assets/204a0806-5623-4fbe-a1e7-6f150279ea1a

# News :triangular_flag_on_post:
- [2025/03/17] **Code of HoloGest release** ⭐
- [2024/12/15] **DIDiffGes got accepted by AAAI 2025!** 🎉
- [2024/11/10] **HoloGest got accepted by 3DV 2025!** 🎉


## 1. Getting started

This code was tested on `NVIDIA GeForce RTX 3070 Ti` and requires:

* conda3 or miniconda3

```
cd ./main/
bash pip_install.sh
```

## 2. Pre-trained model and data
- Download [CLIP](https://drive.google.com/drive/folders/1CN9J2T1tN-F2R5qfHjOfMkGXP00oka6E?usp=drive_link) model , [ASR](https://drive.google.com/drive/folders/1tvQQp6vacDcPg5T6WZIgWjvrk46nYin4?usp=sharing)and pre-trained weights from [here](https://drive.google.com/file/d/14kH1QGBHaMLsdPIrNvEbgNnc32vfHK4U/view?usp=drive_link). Put all the folder in `./main/holgest/`.
- Download [TextEncoder](https://drive.google.com/drive/folders/1J39SDT3RwMH7v7dJl_53stb0wkwTbXEY?usp=drive_link) and put it in `./main/model/`
- Download WavLM weights from [here](https://drive.google.com/drive/folders/1du41ziM0utAMjCtn-YPM8ZYOI6YplHrq?usp=drive_link) and put it in `./main/hologest`


## 3. Quick Start

```
bash demo.sh
```

## Applications

https://github.com/user-attachments/assets/9ca21aab-e7ed-4326-80fc-23d2c1110f3c

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
