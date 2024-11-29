
# LiDAR_UDA

This repo focuses on self-training-based methods for 3D outdoor driving scenario LiDAR point clouds UDA segmentation.

Currently, this repo includes the implementation of [LaserMix](https://arxiv.org/abs/2207.00026), [SAC-LM](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Density-guided_Translator_Boosts_Synthetic-to-Real_Unsupervised_Domain_Adaptive_Segmentation_of_3D_CVPR_2024_paper.pdf) and [CoSMix](https://arxiv.org/abs/2207.09778).

## Updates

- \[2024.12\] - Added support for the CoSMix.

## TODO List
- [x] Implementation of CoSMix
- [x] Upload tranied models
- [x] Implementation of SAC-LM
- [x] Implementation of LaserMix

## Getting Started
```Shell
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda

# pytorch
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# MinkowskiEngine==0.5.4
# make sure your gcc is below 11
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

pip install tensorboard
pip install setuptools==52.0.0
pip install six
pip install pyyaml
pip install easydict
pip install gitpython
pip install wandb
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install tqdm
pip install pandas
pip install scikit-learn
pip install opencv-python
```
pip install other packages if needed.

Our released implementation is tested on
+ Ubuntu 18.04
+ Python 3.8 
+ PyTorch 1.10.1
+ MinkowskiEngine 0.5.4
+ NVIDIA CUDA 11.1
+ 3090 GPU

## DGT preparing
Please refer to [DGT.md](docs/DGT.md) for the details.

## Train
1. Please train a SourceOnly model (see [DGT-ST](https://github.com/yuan-zm/DGT-ST/tree/master/SourceOnly_DGTST)) or directly download the [pretrained model](#model_zoo) and organize the downloaded files as follows
```
LiDAR_UDA
├── preTraModel
│   ├── Syn2SK
│   │   │── SourceOnly
│   │   │── stage_1_PCAN
│   │   │── stage_2_SAC_LM
│   ├── Syn2Sp
│   │   │── SourceOnly
│   │   │── stage_1_PCAN
│   │   │── stage_2_SAC_LM
├── change_data
├── configs
```

#### Important notes: Please choose `PRETRAIN_PATH` carefully in the `config file`.

SynLiDAR -> SemanticKITTI:

### LaserMix

Follow the experimental settings of CoSMix and use `checkpoint_epoch_10.tar` as the pre-trained model.


``` python run_experiments.py --cfg=configs/SynLiDAR2SemanticKITTI/LaserMix.yaml```

### SAC-LM
We use `checkpoint_val_target_Sp.tar`, the model pretrained by PCAN, as the pre-trained model.

``` python run_experiments.py --cfg=configs/SynLiDAR2SemanticKITTI/stage_2_SAC_LM.yaml```


## Test

For SynLiDAR $\rightarrow$ semanticKITTI,

<details><summary>Code</summary>

```
python infer.py \
        --checkpoint_path preTraModel/Syn2SK/SynLiDAR2SK_M34_XYZ/2022-10-25-13_42/checkpoint_epoch_10.tar \
        --result_dir res_pred/syn2sk/SynLiDAR2SK_M34_XYZ_Epoch10 \
        --batch_size 8 \
        --num_classes 20 \
        --domain target \
        --cfg configs/SynLiDAR2SemanticKITTI/stage_2_SAC_LM.yaml

python eval.py \
        --dataset ~/dataset/semanticKITTI/dataset/sequences \
        --predictions res_pred/syn2sk/SynLiDAR2SK_M34_XYZ_Epoch10 \
        --sequences 08 \
        --num-classes 20 \
        --dataset-name SemanticKITTI \
        --datacfg dataset/configs/SynLiDAR2SemanticKITTI/semantic-kitti.yaml

```
</details>


For SynLiDAR $\rightarrow$ semanticPOSS,
<details><summary>Code</summary>

```
python infer.py \
        --checkpoint_path preTraModel/Syn2Sp/SynLiDAR2SP_M34_XYZ/2022-10-27-08_58/checkpoint_epoch_10.tar \
        --result_dir res_pred/syn2sp/SynLiDAR2SP_M34_XYZ_Epoch10 \
        --batch_size 8 \
        --num_classes 20 \
        --domain target \
        --cfg configs/SynLiDAR2SemanticPOSS/stage_2_SAC_LM.yaml

python eval.py \
        --dataset ~/dataset/semanticPOSS/dataset/sequences/ \
        --predictions res_pred/syn2sp/SynLiDAR2SP_M34_XYZ_Epoch10 \
        --sequences 03 \
        --num-classes 14 \
        --dataset-name SemanticPOSS \
        --datacfg dataset/configs/SynLiDAR2SemanticPOSS/semantic-poss.yaml
```
</details>


## Model Zoo <a id="model_zoo"></a>


We release the checkpoints of SynLiDAR -> SemanticKITTI and SynLiDAR -> SemanticPOSS. You can directly use the provided model for testing. Then, you will get the same results as in Tab.1 and Tab. 2 in our [paper](https://arxiv.org/pdf/2403.18469.pdf).

#### Important notes: The provided checkpoints are all trained by our **unclean** code.

[Baidu (提取码：btod)](https://pan.baidu.com/s/1yPuNvFnDPnd9lBF-7I6UHw?pwd=btod) 

[Google](https://drive.google.com/drive/folders/1EuxDphixI579hBgiOQFoVz5KlVwHllkP?usp=sharing)

### Evaluation Results on SemanticKITTI

| Method | mIoU | car | bicycle | motorcycle | truck | other-vehicle | person | bicyclist | motorcyclist | road | parking | sidewalk | other-ground | building | fence | vegetation | trunk | terrain | pole | traffic-sign |
|-------- |-----|-----|----------|------------|-------|---------------|--------|-----------|--------------|------|---------|-----------|--------------|-----------|-------|------------|--------|----------|------|--------------|
| [CoSMix](https://drive.google.com/drive/folders/1LunRRxN2mrVJXfRPHYsLkkY_2uZ8hFyI) | 30.1 | 79.3 | 7.4 | 8.5 | 7.9 | 14.1 | 23.2 | 31.5 | 1.6 | 63.7 | 11.3 | 38.3 | 0.3 | 63.7 | 13.5 | 73.6 | 47.5 | 20.3 | 43.8 | 22.1 |
| [LaserMix](https://drive.google.com/drive/folders/1IgWmfU3UtMo9hN8jS1FNbQzo42vQWt53) | 36.0 | 86.7 | 9.3 | 33.3 | 3.0 | 3.4 | 40.3 | 57.2 | 5.0 | 75.9 | 11.5 | 54.0 | 0.0 | 60.4 | 8.6 | 77.3 | 45.9 | 49.2 | 44.6 | 17.5 |
| [SAC-LM](https://drive.google.com/drive/folders/12RN3Os0WMLyJMLfqUsWAsyfd_DxUhFh4) | 43.1 | 92.9 | 17.3 | 43.4 | 15.0 | 6.1 | 49.2 | 54.2 | 4.2 | 86.4 | 19.1 | 62.3 | 0.0 | 78.2 | 9.2 | 83.3 | 56.0 | 59.1 | 51.2 | 32.3 |

### Evaluation Results on SemanticPOSS

| Method | mIoU | rider | car | trunk | plant | traffic-sign | pole | trashcan | building | cone_stone | fence | bike | ground | person |
|--------|------|-------|-----|-------|-------|--------------|------|-----------|-----------|------------|-------|------|---------|---------|
| [CosMix](https://drive.google.com/drive/folders/1xVr9Iw0KmU5_yBxPNy-7Np5tHkzzXt1I) | 44.4 | 42.1 | 34.0 | 45.2 | 63.9 | 39.9 | 35.6 | 2.3 | 67.4 | 18.8 | 42.9 | 44.4 | 79.5 | 61.6 |
| [LaserMix](https://drive.google.com/drive/folders/1w6_DYW3v9nKd8LdoWUoLhiu3wyLmZK74) | 46.0 | 58.1 | 59.8 | 48.8 | 69.0 | 23.6 | 38.7 | 32.6 | 59.5 | 14.8 | 42.7 | 9.0 | 79.4 | 61.9 |
| [SAC-LM](https://drive.google.com/drive/folders/1S0Gt8PI7YRjfyzpanF1lmrfzay-tAljI) | 50.8 | 55.1 | 70.7 | 46.1 | 74.2 | 30.1 | 36.3 | 44.1 | 81.0 | 4.3 | 62.8 | 10.3 | 78.5 | 67.2 |

## Acknowledgement
Thanks for the following works for their awesome codebase.

[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

[SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)

[CoSMix](https://github.com/saltoricristiano/cosmix-uda)

[LaserMix](https://github.com/ldkong1205/LaserMix)

[LiDAR Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990175.pdf)


## Citation

```
@inproceedings{DGTST,
    title={Density-guided Translator Boosts Synthetic-to-Real Unsupervised Domain Adaptive Segmentation of 3D Point Clouds},
    author={Zhimin Yuan, Wankang Zeng, Yanfei Su, Weiquan Liu, Ming Cheng, Yulan Guo, Cheng Wang},
    booktitle={CVPR},
    year={2024}
}
```