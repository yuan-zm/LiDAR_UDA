
# syn2sk
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

# syn2sp
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