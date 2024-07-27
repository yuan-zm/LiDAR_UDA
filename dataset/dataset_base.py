from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import torch.utils.data as torch_data
import torch
import yaml
import os
import copy

from .data_utils import get_sk_data, augment_noisy_rot, augment_rot_z
import MinkowskiEngine as ME
from dataset.downsample_utils import get_specific_beam_mask


class SemanticDataset(torch_data.Dataset):
    def __init__(self, data_cfg, mode):
        
        self.data_cfg = data_cfg 
        self.name = data_cfg.TYPE 
        # if mode == 'train':
        #     assert self.d_domain == data_cfg.d_domain, "Error domain for initialization."
        # elif mode == 'validation' or mode == 'test' or mode == 'gen_pselab':
        #     assert self.d_domain == data_cfg.d_domain+'_infer', "Error domain for initialization."

        self.dataset_path = os.path.expanduser(data_cfg.DATASET_DIR)

        # for sparse
        self.quantization_size = data_cfg.VOXEL_SIZE
        self.in_num_voxels = data_cfg.IN_NUM_VOXELS 
        
        # load data mapping file
        DATA = yaml.safe_load(open(data_cfg.data_config_file, 'r'))
        if data_cfg.class_mapping is not None:
            remap_dict = DATA[data_cfg.class_mapping]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        else:
            self.remap_lut = None
        self.label_name = DATA[data_cfg.class_mapping_labels]
        
        # load all data list
        self.mode = mode
        if mode == 'train' or mode == 'gen_pselab':
            seq_list = data_cfg.train_seq_list
        elif mode == 'validation' or mode == 'test':
            seq_list = data_cfg.valid_seq_list
        self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        self.data_list = sorted(self.data_list)

        print('This is ** {} ** dataset, filepath ** {} ** mode is ** {} **, has ** {} ** scans.'.
                format(self.name, self.dataset_path, self.mode, len(self.data_list)))
        
        # for augmentation
        self.noisy_rot = 0.1
        self.flip_y = 0.5
        self.rot_z = 6.2831  # 2 * pi

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes, self.remap_lut)

    def __len__(self):
        if self.is_debug:
            return 20
        else:
            return len(self.data_list)
            
    def __getitem__(self, item):
        return None

    def gen_sample(self, cloud_ind):  # Generator loop

        pc_name = self.data_list[cloud_ind]
        # get one frame sample
        slt_pc, slt_remis, slt_lab = get_sk_data(pc_name, self.dataset_path, self.remap_lut, self.name)
        
        # augmentation
        if self.mode == 'train':
            if np.random.random() > 0.5:
                slt_pc = augment_noisy_rot(slt_pc, noisy_rot=self.noisy_rot)
            if np.random.random() > 0.5:
                slt_pc = augment_rot_z(slt_pc, rot_z=self.rot_z)
        
        # construct a sparse Tensor
        if self.data_cfg.USE_INTENSITY and self.data_cfg.DATASET_TARGET.USE_INTENSITY:
            feature = np.concatenate((slt_pc, slt_remis.reshape(-1, 1)), axis=1)
        v_coords, v_ft, v_lab, uni_map, inv_map = ME.utils.sparse_quantize(
                                                    coordinates=slt_pc,
                                                    features=feature if self.data_cfg.USE_INTENSITY else slt_pc,
                                                    labels=slt_lab,
                                                    quantization_size=self.quantization_size,
                                                    return_index=True,
                                                    return_inverse=True)
        v_remis = slt_remis[uni_map]

        # choose fix input num of voxels
        slt_idx = None
        if self.mode == 'train' and self.in_num_voxels > 0:
            if len(v_coords) > self.in_num_voxels:
                slt_idx = np.random.choice(len(v_coords), self.in_num_voxels, replace=False)
                v_coords = v_coords[slt_idx]
                v_ft = v_ft[slt_idx]
                v_lab = v_lab[slt_idx]
                v_remis = v_remis[slt_idx]

        v_lab[v_lab == -100] = 0
        cloud_ind = np.array([cloud_ind], dtype=np.int32)

        if self.d_domain == 'target':  # set slt_lab as 0
            slt_lab = np.zeros_like(slt_lab)
            v_lab = np.zeros_like(v_lab)

        sp_data = (v_coords, v_ft, v_lab, inv_map, uni_map, v_remis)
        
        data_dict = {} # put all data into a dict 
        data_dict['slt_pc'] = slt_pc
        data_dict['slt_lab'] = slt_lab
        data_dict['slt_idx'] = slt_idx
        data_dict['cloud_ind'] = cloud_ind
        data_dict['sp_data'] = sp_data
        
        # --------------  DGT ---------------
        if self.use_dgt:
            aug_sp_data = self.Density_uided_Translator(pc_name, v_ft, v_lab, v_remis, uni_map)
            data_dict['aug_sp_data'] = aug_sp_data
        else:
            del_mask = np.ones_like(v_lab)
            aug_sp_data = (v_coords, v_ft, v_lab, inv_map, uni_map, v_remis, del_mask)

        return data_dict

    def Density_uided_Translator(self, pc_name, v_ft, v_lab, v_remis, uni_map): 
        
        drop_beam = False
        if self.src_total_beams > self.tgt_total_beams and self.d_domain == 'source':
            drop_beam = True
        if self.src_total_beams < self.tgt_total_beams and self.d_domain == 'target':
            drop_beam = True
            
        # construct DGT voxel representation 
        if drop_beam and \
                self.mode == 'train': #  and np.random.random() > 0.5
            
            beamLabel_path = os.path.expanduser(self.data_cfg.DGT.beam_label_path)
            
            seq_id, frame_id = pc_name[0], pc_name[1]
            beamLabel_inds_path = join(beamLabel_path, seq_id, frame_id + '.npy')
            beamLabel = np.load(beamLabel_inds_path)

            total_beam_labels = np.arange(0, self.src_total_beams)
            
            if self.tgt_dataset_name == 'SemanticPOSS': # 40 beams
                choose_beams = total_beam_labels[::2] # chose 32 beams first
                rest_choose_beams = total_beam_labels[1::2][:8] # chose another 8 beams near the LiDAR center
                choose_beams = np.concatenate((choose_beams, rest_choose_beams), axis=0)
            if self.tgt_dataset_name == 'nuScenes': # 32 beams
                choose_beams = total_beam_labels[::2] # only use 32 beams

            # 得到保留点下的线上点 True 这条线上的点是保留的 False 这条线上的点是删除的
            choosed_specificBeamPc_mask = get_specific_beam_mask(beamLabel, choose_beams)
            choosed_specificBeamVoxel_mask = choosed_specificBeamPc_mask[uni_map]
        else:
            choosed_specificBeamVoxel_mask = None
            
        if np.random.random() > self.aug_data_prob:
            aug_sp_data = self.range_drop(v_ft, v_lab, v_remis, choosed_specificBeamVoxel_mask)
            
        return aug_sp_data
    
    def range_drop(self, pc, lab, remis, saveVoxel_mask):
        
        del_mask = np.ones_like(lab)
        if saveVoxel_mask is None:
            del_inds = []
        else:
            del_inds = np.where(saveVoxel_mask == 0)[0].tolist()
            
        if self.d_domain == 'source' and self.mode == 'train': #  and np.random.random() > 0.5
            src2tgt_denseity_raio = self.tgt_density / (self.src_density + 1e-10)
            drop_prob = 1 - np.clip(src2tgt_denseity_raio, a_min=0, a_max=1.)
        else:
            tgt2src_denseity_raio = self.src_density / (self.tgt_density + 1e-10)
            drop_prob = 1 - np.clip(tgt2src_denseity_raio, a_min=0, a_max=1.)

        xy_dis = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        dis_range = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float) # np.linspace(0, 100, 10)
        # dis_range = np.linspace(0, 100, 10)
        for i in range(10):
            if drop_prob[i] > 0:
                if saveVoxel_mask is None:
                    this_range_ind = np.where((xy_dis > dis_range[i]) & (xy_dis < dis_range[i+1]))[0] 
                else:
                    this_range_ind = np.where((xy_dis > dis_range[i]) & (xy_dis < dis_range[i+1]) & (saveVoxel_mask == True))[0] 
                this_del_num = int(len(this_range_ind) * drop_prob[i])
                this_drop_ind = np.random.choice(this_range_ind, this_del_num, replace=False)
                del_inds.extend(this_drop_ind)
                
        aug_drop_pc = np.delete(copy.deepcopy(pc), del_inds, axis=0)
        aug_drop_remis = np.delete(copy.deepcopy(remis), del_inds, axis=0)
        aug_drop_lab = np.delete(copy.deepcopy(lab), del_inds, axis=0)
        del_mask[del_inds] = 0

        # add shift
        if np.random.random() > self.shift_prob and self.d_domain == 'source':
            N, C = aug_drop_pc.shape
            shift_range = self.shift_range #  0.1 # 05
            assert(shift_range > 0)
            shifts = np.random.uniform(-shift_range, shift_range, (N, C))

            end_shifts = np.zeros_like(shifts)
            shift_chose_ind = np.random.choice(shifts.shape[0], int(shifts.shape[0]* 0.1), replace=False)
            end_shifts[shift_chose_ind] = shifts[shift_chose_ind]
            aug_drop_pc[:, :2] += end_shifts[:, :2]

        aug_v_coords, aug_v_ft, aug_v_lab, aug_uni_map, aug_inv_map = ME.utils.sparse_quantize(
                                                                                                coordinates=aug_drop_pc,
                                                                                                features=aug_drop_pc,
                                                                                                labels=aug_drop_lab,
                                                                                                quantization_size=self.quantization_size,
                                                                                                return_index=True,
                                                                                                return_inverse=True)
        aug_v_remis = aug_drop_remis[aug_uni_map]

        aug_v_lab[aug_v_lab == -100] = 0

        aug_sp_data = (aug_v_coords, aug_v_ft, aug_v_lab, aug_inv_map, aug_uni_map, aug_v_remis, del_mask)
        
        return aug_sp_data
    
    
    def collate_fn(self, data):
        slt_pc = [torch.from_numpy(d['slt_pc']) for d in data]
        slt_lab = [torch.from_numpy(d['slt_lab']) for d in data]
        
        cloud_ind = [torch.from_numpy(d['cloud_ind']) for d in data]
        cloud_ind = torch.cat(cloud_ind, 0).long()
        
        slt_lab = [torch.from_numpy(d['slt_lab']) for d in data]
        slt_lab = torch.cat(slt_lab, 0).long()
        
        sp_data = [d['sp_data'] for d in data]
      
        if self.use_dgt and 'infer' not in self.d_domain:
            aug_sp_data = [d['aug_sp_data'] for d in data]
     
        inputs = {}
        inputs['pc_labs'] = slt_lab
        inputs['cloud_inds'] = cloud_ind
        
        # original input
        coords, feats, labels, inverse_map, unique_map, sp_remis = list(zip(*sp_data))
        inputs['sp_remis'] =  torch.from_numpy(np.concatenate(sp_remis, 0)).float()
        # Generate batched coordinates
        inputs['coords_mink'] = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        inputs['feats_mink'] = torch.from_numpy(np.concatenate(feats, 0)).float()
        inputs['labels_mink'] = torch.from_numpy(np.concatenate(labels, 0)).long()
        orig_inv_maps, orig_uni_maps  = self.get_inv_unq_map(inverse_map, unique_map) 
        inputs['inverse_map'], inputs['unique_map'] = orig_inv_maps, orig_uni_maps 

        if self.use_dgt:
            aug_coords, aug_feats, aug_labels, aug_inverse_map, aug_unique_map, aug_sp_remis, del_mask = list(zip(*aug_sp_data))
            inputs['aug_del_mask'] = torch.from_numpy(np.concatenate(del_mask, 0)).bool()
            inputs['aug_sp_remis'] =  torch.from_numpy(np.concatenate(aug_sp_remis, 0)).float()
    
            inputs['aug_coords_mink'] = ME.utils.batched_coordinates(aug_coords)
            inputs['aug_feats_mink'] = torch.from_numpy(np.concatenate(aug_feats, 0)).float()
            inputs['aug_labels_mink'] = torch.from_numpy(np.concatenate(aug_labels, 0)).long()

            aug_orig_inv_maps, aug_orig_uni_maps = self.get_inv_unq_map(aug_inverse_map, aug_unique_map) 
            inputs['aug_inverse_map'], inputs['aug_unique_map'] = aug_orig_inv_maps, aug_orig_uni_maps
           
        return inputs
    
    def get_inv_unq_map(self, inverse_map, unique_map):
        list_inverse_map = list(inverse_map)
        list_unique_map = list(unique_map)
        post_len, unique_len = 0, 0

        for i_list in range(len(list_inverse_map)):
            list_inverse_map[i_list] = list_inverse_map[i_list] + post_len
            post_len += unique_map[i_list].shape[0]

            list_unique_map[i_list] = list_unique_map[i_list] + unique_len
            unique_len += inverse_map[i_list].shape[0]
        
        return torch.from_numpy(np.concatenate(list_inverse_map, 0)).long(), torch.from_numpy(np.concatenate(list_unique_map, 0)).long()
        