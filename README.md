# URFusion
Code of "URFusion: Unsupervised Unified Degradation-Robust Image Fusion Network" (TIP 2025).

## Introduction:
This study proposes an unsupervised unified degradation-robust image fusion network, termed as URFusion, for visible and infrared image fusion and multi-exposure image fusion. In this work, various types of degradations can be uniformly eliminated during the fusion process in an unsupervised manner. It is composed of three core modules:<br>
i) intrinsic content extraction (extract degradation-free intrinsic content features from images affected by various degradations);<br>
ii) intrinsic content fusion (with content features to provide feature-level rather than image-level fusion constraints for optimizing the content fusion network, eliminating degradation residues and reliance on ground truth);<br>
iii) appearance representation learning and assignment (learn the appearance representation of images and assign the statistical appearance representation of high-quality images to the content-fused result, producing the final high-quality fused image).
<br>

The framework of this method is shown below:
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/URFusion_framework.jpg" width="870" height="416"/></div>
<br>


## Recommended Environment:
python=3.6<br>
tensorflow-gpu=1.14.0<br>
numpy=1.19<br>
scikit-image=0.17.2<br>
pillow=8.2<br>

### __To train__:
* __Training dataset:__
  *  Download the training data: [LOL](https://daooshee.github.io/BMVC2018website/), [AGLIE](https://phi-ai.buaa.edu.cn/project/AgLLNet/index.htm), and [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) datasets.
  * Select part of the data for training, and put the low-light images and corresponding normal-light images in `./dataset/low/` and `./dataset/high/`, respectively.
  * Can also put a small number of paired low-light and normal-light images in `./dataset/eval/low/` and `./dataset/eval/high/` for validation during the training phase.

* __Train the decomposition network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_decomposition_network.py```<br>
  * The relevant files are stored in `./checkpoint/decom/`, `./logs/decom/`, and `./eval_result/decom/`

* __Train the color shift estimation network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_color_network.py```<br>
  * The relevant files are stored in `./checkpoint/color_net/`, `./logs/color_net/`, and `./eval_result/color/`

* Train the spatially variant pollution estimation network:<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_noise_network.py```<br>
  * The relevant files are stored in `./checkpoint/noise_net/`, `./logs/noise_net/`, and `./eval_result/noise/`

* __Train the illumination adjustment network:__<br>
  * Run ```CUDA_VISIBLE_DEVICES=0 python train_illu_adjust_network.py```<br>
  * The relevant files are stored in `./checkpoint/illu_adjust/`, `./logs/illu_adjust/`, and `./eval_result/illu_adjust/`

### To test:
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

