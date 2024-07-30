
# LiDAR_UDA

This repo focuses on self-training-based methods for 3D outdoor driving scenario LiDAR point clouds UDA segmentation.

Currently, this repo includes the implementation of [LaserMix](https://arxiv.org/abs/2207.00026) and [SAC-LM](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Density-guided_Translator_Boosts_Synthetic-to-Real_Unsupervised_Domain_Adaptive_Segmentation_of_3D_CVPR_2024_paper.pdf).

## TODO List
- [ ] Implementation of CoSMix
- [ ] Upload tranied models
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