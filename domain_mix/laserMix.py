import numpy as np

import torch    
import torch.nn.functional as F
import MinkowskiEngine as ME

import collections

# for vis
from utils.ply_writer import write_ply
from utils.common import syn2sk_color_dic
field_names = ['x', 'y', 'z', 'red', 'green', 'blue']

class LaserMix:
    def __init__(self, cfg):
        self.cfg = cfg
        
        assert cfg.DATASET_SOURCE.VOXEL_SIZE == cfg.DATASET_TARGET.VOXEL_SIZE, 'voxel size error.'
        self.voxel_size = cfg.DATASET_SOURCE.VOXEL_SIZE
        
        self.batch_size = cfg.DATALOADER.TRA_BATCH_SIZE
        
        # Actually, this is the pitch angle range for a specific domain.
        # However, we do not change this value for simplicity.
        pitch_angles = [-25, 3]
        self.pitch_angle_down = pitch_angles[0] / 180 * np.pi
        self.pitch_angle_up = pitch_angles[1] / 180 * np.pi
        # Please refer to the LaserMix paper for the details i.e., Table 4 in the paper.
        self.num_areas = [3, 4, 5, 6] 

        if cfg.DATASET_SOURCE.get("use_dgt_for_laserMix", False):
            self.src_pc_prefix = 'aug_'
        else:
            self.src_pc_prefix = ''

        if cfg.DATASET_TARGET.get("use_dgt_for_laserMix", False):
            self.tgt_pc_prefix = 'aug_'
        else:
            self.tgt_pc_prefix = ''
            
    def mix(self, src_BData, tgt_BData):
        # obtain source data
        b_src_vos = src_BData[self.src_pc_prefix + 'coords_mink']
        b_src_lbs = src_BData[self.src_pc_prefix + 'labels_mink']
        b_src_fts = src_BData[self.src_pc_prefix + 'feats_mink']
        
        # obtain target data
        b_tgt_vos = tgt_BData[self.tgt_pc_prefix + 'coords_mink']        
        # get the pseudo label for target pcs
        if tgt_BData.get('pseudo_label', None) is not None:
            b_tgt_lbs = tgt_BData['pseudo_label']
        else:
            b_tgt_lbs = torch.zeros(b_tgt_vos.shape[0]).cuda()
        b_tgt_fts = tgt_BData[self.tgt_pc_prefix + 'feats_mink']
        
        assert b_src_vos[-1, 0] + 1 == self.batch_size, 'batch size error.'
        
        mixed_batch = {
                     'mixed_coors_1': [],
                     'mixed_lbls_1': [],
                     'mixed_feats_1': [],
                     'mixed_masks_1': [],
                     'mixed_list_uniMap_1': [],
                     'mixed_list_invMap_1': [],
                     'mixed_uniMap_1': [],
                     'mixed_invMap_1': []
                        }
        
        for b in range(self.batch_size):
            src_b_idx = b_src_vos[:, 0] == b
            tgt_b_idx = b_tgt_vos[:, 0] == b

            # source
            src_pt = b_src_vos[src_b_idx, 1:] * self.voxel_size # return voxel to pcs
            src_lb = b_src_lbs[src_b_idx]
            src_ft = b_src_fts[src_b_idx]

            # target
            tgt_pt = b_tgt_vos[tgt_b_idx, 1:] * self.voxel_size # return voxel to pcs
            tgt_lb = b_tgt_lbs[tgt_b_idx]
            tgt_ft = b_tgt_fts[tgt_b_idx]

            rho_src = torch.sqrt(src_pt[:, 0]**2 + src_pt[:, 1]**2)
            pitch_src = torch.atan2(src_pt[:, 2], rho_src)
            pitch_src = torch.clamp(pitch_src, self.pitch_angle_down + 1e-5, self.pitch_angle_up - 1e-5)

            rho_tgt = torch.sqrt(tgt_pt[:, 0]**2 + tgt_pt[:, 1]**2)
            pitch_tgt = torch.atan2(tgt_pt[:, 2], rho_tgt)
            pitch_tgt = torch.clamp(pitch_tgt, self.pitch_angle_down + 1e-5, self.pitch_angle_up - 1e-5)

            num_areas = np.random.choice(self.num_areas, size=1)[0]
            angle_list = np.linspace(self.pitch_angle_up, self.pitch_angle_down, num_areas + 1)
            
            src_mask = torch.ones(src_pt.shape[0]) # source mask =1
            tgt_mask = torch.zeros(tgt_pt.shape[0]) # target mask =0
            
            pt_mix_1, pt_mix_2 = [], []
            lb_mix_1, lb_mix_2 = [], []
            ft_mix_1, ft_mix_2 = [], []
            mask_mix_1, mask_mix_2 = [], []
            
            for i in range(num_areas):
                # convert angle to radian
                start_angle = angle_list[i + 1]
                end_angle = angle_list[i]
                idx_src = (pitch_src > start_angle) & (pitch_src <= end_angle)
                idx_tgt = (pitch_tgt > start_angle) & (pitch_tgt <= end_angle)
                if i % 2 == 0:  # pick from original point cloud
                    pt_mix_1.append(src_pt[idx_src])
                    lb_mix_1.append(src_lb[idx_src])
                    ft_mix_1.append(src_ft[idx_src])
                    mask_mix_1.append(src_mask[idx_src])
            
                    pt_mix_2.append(tgt_pt[idx_tgt])
                    lb_mix_2.append(tgt_lb[idx_tgt])
                    ft_mix_2.append(tgt_ft[idx_tgt])
                    mask_mix_2.append(tgt_mask[idx_tgt])

                else:  # pickle from mixed point cloud
                    pt_mix_1.append(tgt_pt[idx_tgt])
                    lb_mix_1.append(tgt_lb[idx_tgt])
                    ft_mix_1.append(tgt_ft[idx_tgt])
                    mask_mix_1.append(tgt_mask[idx_tgt])

                    pt_mix_2.append(src_pt[idx_src])
                    lb_mix_2.append(src_lb[idx_src])
                    ft_mix_2.append(src_ft[idx_src])
                    mask_mix_2.append(src_mask[idx_src])
                    
            mixed_batch = construct_sparse_tensor(pt_mix_2,
                                            lb_mix_2,
                                            ft_mix_2,
                                            mask_mix_2,
                                            self.voxel_size,
                                            b,
                                            mixed_batch,
                                            mix_num="1")   
        
        mixed_invMap_1, mixed_uniMap_1 = get_inv_unq_map(mixed_batch['mixed_list_invMap_1'],
                                                         mixed_batch['mixed_list_uniMap_1'])
        mixed_batch['mixed_invMap_1'], mixed_batch['mixed_uniMap_1'] = mixed_invMap_1, mixed_uniMap_1
            
        for k, i in mixed_batch.items():
           
            if 'Map' not in k:
                mixed_batch[k] = torch.cat(i, dim=0)
        
        return mixed_batch
      

def construct_sparse_tensor(pts, lbls, feats, domain_masks,
                            voxel_size, batch_ind,
                            new_batch, mix_num):
    pts = torch.cat(pts)
    lbls = torch.cat(lbls)
    feats = torch.cat(feats)
    domain_masks = torch.cat(domain_masks)
    
    
    _, _, uni_map, inv_map = ME.utils.sparse_quantize(coordinates=pts,
                                                    features=feats,
                                                    quantization_size=voxel_size,
                                                    return_index=True,
                                                    return_inverse=True)
    v_pts = pts[uni_map]
    v_lbls = lbls[uni_map]
    v_feats = feats[uni_map]
    v_domain_masks = domain_masks[uni_map]
    
    # for visualization
    # vis_points = v_pts.cpu().numpy()
    # vis_labels = v_lbls.cpu().numpy().reshape(-1)
    # colors = [syn2sk_color_dic[vis_labels[i]] for i in range(len(vis_labels))]
    # colors = np.array(colors).astype(np.int32)
    # write_ply('lm_vis.ply', [vis_points, colors[:, 0], colors[:, 1], colors[:, 2]], field_names)

    v_coors = torch.floor(v_pts / voxel_size)
    v_batch_ind = torch.ones([v_pts.shape[0], 1]).cuda() * batch_ind
    
    bv_coors = torch.cat([v_batch_ind, v_coors], dim=-1)

    new_batch[f'mixed_coors_{mix_num}'].append(bv_coors)
    new_batch[f'mixed_lbls_{mix_num}'].append(v_lbls)
    new_batch[f'mixed_feats_{mix_num}'].append(v_feats)
    new_batch[f'mixed_masks_{mix_num}'].append(v_domain_masks)
    new_batch[f'mixed_list_uniMap_{mix_num}'].append(uni_map)
    new_batch[f'mixed_list_invMap_{mix_num}'].append(inv_map)

    return new_batch

def get_inv_unq_map(inverse_map, unique_map):

    list_inverse_map = list(inverse_map)
    list_unique_map = list(unique_map)
    post_len, unique_len = 0, 0

    for i_list in range(len(list_inverse_map)):
        list_inverse_map[i_list] = list_inverse_map[i_list] + post_len
        post_len += unique_map[i_list].shape[0]

        list_unique_map[i_list] = list_unique_map[i_list] + unique_len
        unique_len += inverse_map[i_list].shape[0]
    
    return torch.from_numpy(np.concatenate(list_inverse_map, 0)).long(), torch.from_numpy(np.concatenate(list_unique_map, 0)).long()
    
    