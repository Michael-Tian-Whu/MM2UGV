<!--
 * @Author: WHURS-THC
 * @Date: 2023-07-21 15:49:54
 * @LastEditTime: 2023-07-24 16:00:04
 * @Description: 
-->

# Multi-modal Self-Supervised Learning for Autonomous Situation Awareness of Unmanned Systems

# Abstract

**Visuomotor intelligent agent** use vision signals as input to directly predict the decisions and actions. Because they require a large corpus of labeled data or environment interactions to achieve satisfactory performance, supervised pre-training is often applied to simultaneously cooperating and **training the perception and control modules in an end-to-end fashion and transfer to downstream tasks** such as visual navigation and trajectory prediction.

<div align="center">
  <img src="figs\intro.drawio.png" width="500"> 
</div>

Using *" intelligent agent self-driving in extraterrestrial planetary environments"* as a case study, the supervised pre-training paradigm suffers from a lack of labeled data and high cost, and inefficient transfer. **Dominant self-supervised approaches in computer vision are not applicable** due to the lack of translation and view invariance in vision-driven driving tasks, and the input contains irrelevant information for driving. Therefore, the research goal is to **design a self-supervised pre-training method applicable to self-driving in extraterrestrial planetary open environments**.



# Method
Inspired by multimodal learning, we introduce **temporal signals such as IMU and Odometry** to help the visual encoder learning. The visual modality is the objective condition for driving decisions, and the temporal signal modality responds to the driving state and decision quality. The two are **synergistic and complementary: the strong correlation between modalities makes it theoretically possible to predict semantic information from one modality to the other, while the inherent differences make cross-modal prediction a more challenging and valuable pretext task compared to within-modality learning**.

 We propose a pre-training method for cross-modal prediction by extracting features of both modalities through a visual encoder and a temporal signal encoder, constructing pseudo-labels of the other modality by clustering the features using the scalable K-Means algorithm, and optimizing the model by repeating the clustering and classification tasks.
<div align="center">
  <img src="figs\arch.drawio.png" width="500"> 
</div>
<!-- ![arch](figs\arch.drawio.png "arch") -->

# Usage

## Prerequisites

The main dependencies are as follows:

- Python 3.8.16
- pytorch 1.12.1
- torchvision 0.13.1
- sklearn 1.2.2
- pillow 9.5.0
- prefetch-generator 1.0.3
- tensorboard 2.12.1
- seaborn 0.12.2

## Self-Supervised Pretraining

This implementation only supports multi-gpu, DataParallel training, which is faster and simpler; single-gpu is also supported but not advised.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 4-gpu machine, run:

```bash  
python main_pretrain.py \
  --gpunum 4 \
  --k 16 \
  --model resnet50 \
  --epoch 4 \
  --subepoch 15 \
  --lr 0.03 \
  --batchsize 32 \
```

To test the pretrained models of both modalities, run: 

```bash  
python main_test.py \
  --gpunum 4 \
  --model resnet50 \
  --dir1 ./v1_k16_epoch15_4_pre/vision_encoder_14 \
  --dir2 ./v1_k16_epoch15_4_pre/sensor_encoder_14 \
```

# Experiment Result

## Few-shot Classification

## Ablation



# Model Zoo