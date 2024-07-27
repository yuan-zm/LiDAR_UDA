
### Syn $\rightarrow$ sk
python domain_mix/cal_cosmix_classWeight.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR \
        --tgt-data-name  SemanticKITTI \
        --data-config-file dataset/configs/SynLiDAR2SemanticKITTI/annotations.yaml \
        --class-mapping map_2_semantickitti \
        --num-classes 20 \
        --voxel-size 0.05

[58532092, 16260531,  1502495,  3794532,
 22976697, 16275388,  10186894, 7542876,
 9952909,  486031109, 3575921,  172770731,
 6871736,  264963005, 39668349, 119917061,
 13146232, 151083513, 14103832, 2560839]

From SynLiDAR to SemanticKITTI sampling_weights:
[0.0,                0.9880716592576068, 0.9988978052239658, 0.9972164211209391,
 0.9831448388154899, 0.9880607605139919, 0.9925271356305252, 0.994466724665657,
 0.9926987814893602, 0.6434590801767024, 0.997376788977194,  0.873259480291243,
 0.9949590570873872, 0.8056294097795189, 0.9709002379098092, 0.9120316818414879,
 0.9903562353053198, 0.8891687101963773, 0.9896537626065551, 0.9981214291108692]

### Syn $\rightarrow$ sp
python domain_mix/cal_cosmix_classWeight.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR \
        --tgt-data-name  SemanticPOSS \
        --data-config-file dataset/configs/SynLiDAR2SemanticPOSS/annotations.yaml \
        --class-mapping map_2_semanticposs \
        --num-classes 14 \
        --voxel-size 0.05

[57135086, 17501326, 49277310, 13147205,
 271024334, 2560968, 14108629, 6594507,
 264980398, 689328, 39674367, 5298938,
 669535051, 10189295]


From SynLiDAR to SemanticPOSS sampling_weights: 
  [0.0,                0.9871745850290106, 0.9638883391233276, 0.9903653951801328,
   0.8013865034691628, 0.9981232577847288, 0.9896608393217328, 0.9951673782429917,
   0.805815652852364,  0.9994948429821191, 0.9709256189795944, 0.9961168040207042,
   0.5093477564672759, 0.9925330265468555]