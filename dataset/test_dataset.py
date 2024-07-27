from os.path import join
import numpy as np
import torch
import MinkowskiEngine as ME

from .dataset_base import SemanticDataset
from .data_utils import get_sk_data

class TestDataset(SemanticDataset):
    def __init__(self, cfg, domain, mode):
        self.d_domain = domain + '_infer'
        
        # for debug mode. Reduce the number of data item for fast debug.
        self.is_debug = cfg.TRAIN.DEBUG
        
        if self.d_domain == 'source_infer':
            data_cfg = cfg.DATASET_SOURCE 
        elif self.d_domain == 'target_infer':
            data_cfg = cfg.DATASET_TARGET
        else:
            raise NotImplementedError("Unkonw domain for initialization.")
        
        self.use_dgt = False
        
        super().__init__(data_cfg, mode) 
        
        self.num_classes = cfg.MODEL_G.NUM_CLASSES
        assert self.num_classes == len(self.label_name), "Error number of classes."
        self.ignored_labels = np.sort([0])
        
    def __getitem__(self, cloud_ind):
        
        pc_name = self.data_list[cloud_ind]
        
        # get one frame sample
        pc, remis, labels = get_sk_data(pc_name, self.dataset_path, self.remap_lut, self.name)
        
        if self.data_cfg.USE_INTENSITY:
            feats = np.concatenate((pc, remis.reshape(-1, 1)), axis=1)
        else:
            feats = pc
            
        sp_coords, sp_feats, sp_lab, unique_map, inverse_map = ME.utils.sparse_quantize(
                                                                coordinates=pc,
                                                                features=feats,
                                                                labels=labels,
                                                                quantization_size=self.quantization_size,
                                                                ignore_label=0,
                                                                return_index=True,
                                                                return_inverse=True
                                                                )

        sp_remis = remis[unique_map]
        sp_lab[sp_lab == -100] = 0
        cloud_ind = np.array([cloud_ind], dtype=np.int32)
        sp_data = (sp_coords, sp_feats, sp_lab, inverse_map, unique_map, sp_remis, len(pc))

        return pc, labels, cloud_ind, sp_data

    def infer_collate_fn(self, batch):
        sp_data = []
        slt_pc, slt_lab, cloud_ind = [], [], []
        
        for i in range(len(batch)):
            slt_pc.append(batch[i][0])
            slt_lab.append(batch[i][1])
            cloud_ind.append(batch[i][2])
            sp_data.append(batch[i][3])
           
        cloud_ind = np.stack(cloud_ind)

        inputs = {}
        inputs['pc_labs'] = torch.from_numpy(np.concatenate(slt_lab)).long()
        inputs['cloud_inds'] = torch.from_numpy(cloud_ind).long()
        
        # get sparse data
        coords, feats, labels, inverse_map, unique_map, sp_remis, s_len = list(zip(*sp_data))
        inputs['s_lens'] = torch.from_numpy(np.stack(s_len, 0)).long()
        inputs['sp_remis'] =  torch.from_numpy(np.concatenate(sp_remis, 0)).float()
    
        # Generate batched coordinates
        inputs['coords_mink'] = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        inputs['feats_mink'] = torch.from_numpy(np.concatenate(feats, 0)).float()
        inputs['labels_mink'] = torch.from_numpy(np.concatenate(labels, 0)).long()
        inputs['inverse_map'], inputs['unique_map'] = self.get_inv_unq_map(inverse_map, unique_map) 

        return inputs
    
    