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
  * Prepare training data: put the training data. i.e., paired visible images of the same scene (same images or images of different degradations), in `./dataset/train/source1/` and `./dataset/train/source2/`, respectively
  * Train the intrinsic content extractor: `cd code` and run `python train_content_extractor.py`<br>
  * Relevant files are stored in `./train-jobs/log/content-extractor/` and `./train-jobs/ckpt/content-extractor_ckpt.pth`
* __2. Train the intrinsic content fusion network:__<br>
  * `cd code` and run `python train_content_fusion.py`<br>
  * Relevant files are stored in `./train-jobs/log/content-fusion/` and `./train-jobs/ckpt/content-fusion_ckpt.pth`
* __3. Train the appearance representation network:__<br>
  * `cd code` and run `python A2V.py`<br>
  * Relevant files are stored in `./train-jobs/log/A2V/` and `./train-jobs/ckpt/A2V_ckpt.pth`
  * Put some high-quality normal-light images in `./dataset/train/normal_img/`
  * `cd code` and run `centroid.py` to obtain the statistical high-quality appearance representation `./train-jobs/normal_img.mat`<br>

### Visible and Infrared Image Fusion:
*  `cd vis-ir`<br>
* __1. Train the visible intrinsic content extractor as described above (the training data is put in `./dataset/train/VIS/`):__<br>
* __2. Train the infrared intrinsic content extractor:__<br>
  * Prepare training data: put the training data. i.e., paired infrared images of the same scene (same images or images of different degradations), in `./dataset/train/IR/`
  * Train the infrared intrinsic content extractor: `cd code` and run `python train_content_extractor_ir.py`<br>


## __To test:__
### Multi-exposure Image Fusion:
* `cd multi-exposure`<br>
* Put the test data in `./dataset/test/source1/` and `./dataset/test/source2/`
* Run `python test.py`<br>

### Visible and Infrared Image Fusion:


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

