# RSIC-GMamba

<p align="center">
  <img src="images/RSIC-GMamba.png" alt="RSIC-GMamba" width="800"/>
</p>

## Installation and Dependencies
Create the `gmamba` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate gmamba
```
## Data preparation
For the evaluation metrics, Please download the [evaluation.zip](https://pan.baidu.com/s/13ZfH-CMYbW3RsW0-RX7KKQ)(BaiduPan code:wuiu) and extract it to `./evaluation`.


For Feature Extraction:  Please refer to [pre_CLIP_feature](https://github.com/One-paper-luck/MG-Transformer/blob/main/feature_pro/pre_CLIP_feature.py)


## Train
```
python train.py
```

## Evaluate
```
python test.py
```


# Citation:
```
@ARTICLE{,
  author={Meng, Lingwu and Wang, Jing and Xiao, Liang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Multiscale Grouping Transformer with CLIP Latents for Remote Sensing Image Captioning}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}}
```



## Reference:
1. https://github.com/tylin/coco-caption
2. https://github.com/aimagelab/meshed-memory-transformer
