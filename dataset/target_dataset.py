from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import yaml
import os
from .dataset_base import SemanticDataset


class TargetDataset(SemanticDataset):
    def __init__(self, cfg, mode):
        
        # assert cfg.DATASET_TARGET.voxelSizeEnlargeSacles == cfg.DATASET_SOURCE.voxelSizeEnlargeSacles, "Error: voxelSizeEnlargeSacles"
        assert cfg.DATASET_TARGET.USE_INTENSITY == cfg.DATASET_SOURCE.USE_INTENSITY, "Error: USE_INTENSITY"
        assert cfg.DATASET_TARGET.VOXEL_SIZE == cfg.DATASET_SOURCE.VOXEL_SIZE, "Error: VOXEL_SIZE"
        
        self.d_domain = 'target'
        
        # for debug mode. Reduce the number of data item for fast debug.
        self.is_debug = cfg.TRAIN.DEBUG
        
        self.use_dgt = cfg.DATASET_TARGET.get("USE_DGT", False)
        if self.use_dgt:
            self.src_total_beams = cfg.DATASET_SOURCE.DGT.total_beams
            self.tgt_total_beams = cfg.DATASET_TARGET.DGT.total_beams

            self.shift_prob = cfg.DATASET_SOURCE.DGT.aug_shift_prob
            self.shift_range = cfg.DATASET_SOURCE.DGT.aug_shift_range
            self.aug_data_prob = cfg.DATASET_SOURCE.DGT.aug_data_prob
            
            self.src_density = np.array(cfg.DATASET_SOURCE.DGT.DENSITY)
            self.tgt_density = np.array(cfg.DATASET_TARGET.DGT.DENSITY)
            
        super().__init__(cfg.DATASET_TARGET, mode) 
        
        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        assert self.num_classes == len(self.label_name), "Error number of classes."
        self.ignored_labels = np.sort([0])
        
    def __getitem__(self, item):

        return self.gen_sample(item)

    