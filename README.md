# URFusion
Code of "URFusion: Unsupervised Unified Degradation-Robust Image Fusion Network" (TIP 2025).

## Introduction:
This study proposes an unsupervised unified degradation-robust image fusion network, termed as URFusion, for visible and infrared image fusion and multi-exposure image fusion. In this work, various types of degradations can be uniformly eliminated during the fusion process in an unsupervised manner. It is composed of three core modules:<br>
i) Intrinsic content extraction (extract degradation-free intrinsic content features from images affected by various degradations);<br>
ii) Intrinsic content fusion (with content features to provide feature-level rather than image-level fusion constraints for optimizing the content fusion network, eliminating degradation residues and reliance on ground truth);<br>
iii) Appearance representation learning and assignment (learn the appearance representation of images and assign the statistical appearance representation of high-quality images to the content-fused result, producing the final high-quality fused image).
<br>

The framework of this method is shown below:
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/URFusion_framework.jpg" width="870" height="416"/></div>
<br>


## Recommended Environment:
python=3.10<br>
pytorch=1.13<br>
pytorch-cuda=11.7<br>
torchvision=0.14.1<br>
tensorboard=2.14<br>
tensorboardx=2.6.2<br>
numpy=1.24.4<br>
imageio=2.34.2<br>
opencv-python=4.10<br>
pandas=2.0.3<br>
pillow=10.4<br>
scikit-image=0.21
scipy=1.10.1<br>

## __To train:__
### Multi-exposure Image Fusion:
*  `cd multi-exposure`<br>
* __1. Train the visible intrinsic content extractor:__<br>
  * Prepare training data: put the training data. i.e., paired visible images of the same scene (same images or images of different degradations), in `./dataset/train/source1/` and `./dataset/train/source2/`, respectively.
  * Train the intrinsic content extractor: `cd code` and run `python train_content_extractor.py`<br>
  * Relevant files are stored in `./train-jobs/log/content-extractor/` and `./train-jobs/ckpt/content-extractor_ckpt.pth`
* __2. Train the intrinsic content fusion network:__<br>
  * `cd code` and run `python train_content_fusion.py`<br>
  * Relevant files are stored in `./train-jobs/log/content-fusion/` and `./train-jobs/ckpt/content-fusion_ckpt.pth`
* __3. Train the intrinsic content fusion network:__<br>





* __Train the decomposition network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_decomposition_network.py```<br>
  * The relevant files are stored in `./checkpoint/decom/`, `./logs/decom/`, and `./eval_result/decom/`

* __Train the color shift estimation network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_color_network.py```<br>
  * The relevant files are stored in `./checkpoint/color_net/`, `./logs/color_net/`, and `./eval_result/color/`

* Train the spatially variant pollution estimation network:<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_noise_network.py```<br>
  * The relevant files are stored in `./checkpoint/noise_net/`, `./logs/noise_net/`, and `./eval_result/noise/`


## __To test:__
  * Put the test data in `./test_images/`
  * Run ```CUDA_VISIBLE_DEVICES=0 python test.py```<br>
  
If this work is helpful to you, please cite it as:
```
@article{xu2025urfusion,
  title={URFusion: Unsupervised Unified Degradation-Robust Image Fusion Network},
  author={Xu, Han and Yi, Xunpeng and Lu, Chen and Liu, Guangcan and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  volume={34},
  pages={5803--5818},
  publisher={IEEE}
}
```

