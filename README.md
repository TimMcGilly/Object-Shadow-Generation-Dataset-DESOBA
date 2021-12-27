# Object-Shadow-Generation-Dataset-DESOBA


**Object Shadow Generation** is to deal with the shadow inconsistency between the foreground object and the background in a composite image, that is, generating shadow for the foreground object according to background information, to make the composite image more realistic.

Our dataset **DESOBA** is a synthesized dataset for Object Shadow Generation. We build our dataset on the basis of Shadow-OBject Association dataset [SOBA](https://github.com/stevewongv/InstanceShadowDetection),  which  collects  real-world  images  in  complex  scenes  and  provides annotated masks for object-shadow pairs.  Based on SOBA dataset, we remove all the shadows to construct our DEshadowed Shadow-OBject Association(DESOBA) dataset, which can be used for shadow generation task and other shadow-related tasks as well. We illustrate the process of our DESOBA dataset construction based on SOBA dataset in the figure below.

<img src='Examples/task_intro.png' align="center" width=1024>

Illustration of DESOBA dataset construction: The green arrows illustrate the process of acquiring paired data for training and evaluation.  Given a ground-truth target image I<sub>g</sub>, we manually remove all shadows to produce a deshadowed image I<sub>d</sub>. Then, we randomly select a foreground object in I<sub>g</sub>, and replace its shadow area with the counterpart in I<sub>d</sub> to synthesize a composite image I<sub>c</sub> without foreground shadow. I<sub>c</sub> and I<sub>g</sub> form a pair of input composite image and ground-truth target image. 
The red arrow illustrates our shadow generation task. Given I<sub>c</sub> and its foreground mask M<sub>fo</sub>, we aim to generate the target image I<sub>g</sub> with foreground shadow.




 Our DESOBA dataset contains 840 training images with totally 2,999 object-shadow pairs and 160 test images with totally 624 object-shadow pairs. The DESOBA dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1fYqcSjGSr52jppg2LEA1qQ) (access code: sipx), or [**Google Drive**](https://drive.google.com/file/d/114BU47G0OJV3vmx5WKxGnWDSj2Bzh6qS/view?usp=sharing).
 
 <img src='Examples/dataset-samples.png' align="center" width=1024>



# Our SGRNet

Here we provide PyTorch implementation and the trained model of our SGRNet.

## Prerequisites

- Python 
- Pytorch
- PIL

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA.git
cd Object-Shadow-Generation-Dataset-DESOBA
```

- Download the DESOBA dataset.

- We provide the code of obtaining training/testing tuples, each tuple contains foreground object mask, foreground shadow mask, background object mask, background shadow mask, shadow image, and synthetic composite image without foreground shadow mask. The dataloader is available in `/data_processing/data/DesobaSyntheticImageGeneration_dataset.py`, which can be used as dataloader in training phase or testing phase.

- We also provide the code of visualization of training/testing tuple, run:
```bash
python Vis_Desoba_Dataset.py
```
`Vis_Desoba_Dataset.py` is available in `/data_processing/`.
- We show some examples of training/testing tuples in below:
<img src='/data_processing/Visualization_Examples/9.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/5.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/6.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/12.png' align="center" width=1024>
from left to right: synthetic composite image without foreground shadow, target image with foreground shadow, foreground object mask, foreground shadow mask, background object mask, and background shadow mask.

# Data preparing

### 1. Generating training/testing pairs from DESOBA dataset

### 2. Generating real composite testing from test images.

# Shadow Generation Baselines

### 1. Pix2Pix

- Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf).

### 2. Pix2Pix-Res

- Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf). Pix2Pix-Res is a variant of Pix2Pix whose architecture is the same as Pix2Pix but outputs the residual results.

### 3. ShadowGAN

- Image to image translation method. Implementation of paper "*ShadowGAN: Shadow synthesis for virtual objects with conditional adversarial networks*" [[pdf]](https://dc.tsinghuajournals.com/cgi/viewcontent.cgi?article=1127&context=computational-visual-media).



### 4. Mask-ShadowGAN

- Image to image translation method. Implementation of paper "*Mask-ShadowGAN: Learning to remove shadows from unpaired data*" [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Mask-ShadowGAN_Learning_to_Remove_Shadows_From_Unpaired_Data_ICCV_2019_paper.pdf).


### 5. ARShadowGAN

- Image to image translation method. Implementation of paper "*ARShadowGAN: Shadow Generative Adversarial Network for Augmented Reality in Single Light Scenes*" [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ARShadowGAN_Shadow_Generative_Adversarial_Network_for_Augmented_Reality_in_Single_CVPR_2020_paper.pdf).


## Bibtex
If you find this work is useful for your research, please cite our paper using the following **BibTeX  [[arxiv](https://arxiv.org/pdf/2104.10338v1.pdf)]:**

```
@article{hong2021shadow,
  title={Shadow Generation for Composite Image in Real-world Scenes},
  author={Hong, Yan and Niu, Li and Zhang, Jianfu},
  journal={AAAI},
  year={2022}
}
```
