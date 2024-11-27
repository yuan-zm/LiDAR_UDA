import numpy as np

import torch    
import torch.nn.functional as F
import MinkowskiEngine as ME

import collections

# for vis
from utils.ply_writer import write_ply
from utils.common import syn2sk_color_dic
field_names = ['x', 'y', 'z', 'red', 'green', 'blue']

from scipy.linalg import expm, norm

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

class CoSMix:
    def __init__(self, cfg):
        self.cfg = cfg
        
        assert cfg.DATASET_SOURCE.VOXEL_SIZE == cfg.DATASET_TARGET.VOXEL_SIZE, 'voxel size error.'
        self.voxel_size = cfg.DATASET_SOURCE.VOXEL_SIZE
        
        self.batch_size = cfg.DATALOADER.TRA_BATCH_SIZE
        
        self.sampling_weights = np.array(cfg.DATASET_SOURCE.CoSMix_sampling_weights)

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
                        'mixed_coors_src': [],
                        'mixed_lbls_src': [],
                        'mixed_feats_src': [],
                        'mixed_masks_src': [],
                        'mixed_list_uniMap_src': [],
                        'mixed_list_invMap_src': [],
                        'mixed_uniMap_src': [],
                        'mixed_invMap_src': [],
                        
                        'mixed_coors_tgt': [],
                        'mixed_lbls_tgt': [],
                        'mixed_feats_tgt': [],
                        'mixed_masks_tgt': [],
                        'mixed_list_uniMap_tgt': [],
                        'mixed_list_invMap_tgt': [],
                        'mixed_uniMap_tgt': [],
                        'mixed_invMap_tgt': []
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
            
            mix_tgt_pt, mix_tgt_lb, mix_tgt_ft, mix_tgt_mask = mask(origin_pts=src_pt,
                                                                    origin_labels=src_lb,
                                                                    origin_features=src_ft,
                                                                    dest_pts=tgt_pt,
                                                                    dest_labels=tgt_lb,
                                                                    dest_features=tgt_ft,
                                                                    sampling_weights=self.sampling_weights)
            mixed_batch = construct_sparse_tensor(mix_tgt_pt,
                                                  mix_tgt_lb,
                                                  mix_tgt_ft,
                                                  mix_tgt_mask,
                                                  self.voxel_size,
                                                  b,
                                                  mixed_batch,
                                                  mix_num="tgt")   
        
        
            mix_src_pt, mix_src_lb, mix_src_ft, mix_src_mask = mask(origin_pts=tgt_pt,
                                                                    origin_labels=tgt_lb,
                                                                    origin_features=tgt_ft,
                                                                    dest_pts=src_pt,
                                                                    dest_labels=src_lb,
                                                                    dest_features=src_ft,
                                                                    is_pseudo=True)
            
            mixed_batch = construct_sparse_tensor(mix_src_pt,
                                                  mix_src_lb,
                                                  mix_src_ft,
                                                  mix_src_mask,
                                                  self.voxel_size,
                                                  b,
                                                  mixed_batch,
                                                  mix_num="src")
                    
            
        mixed_invMap_src, mixed_uniMap_src = get_inv_unq_map(mixed_batch['mixed_list_invMap_src'],
                                                         mixed_batch['mixed_list_uniMap_src'])
        mixed_batch['mixed_invMap_src'], mixed_batch['mixed_uniMap_src'] = mixed_invMap_src, mixed_uniMap_src
            
        mixed_invMap_tgt, mixed_uniMap_tgt = get_inv_unq_map(mixed_batch['mixed_list_invMap_tgt'],
                                                         mixed_batch['mixed_list_uniMap_tgt'])
        mixed_batch['mixed_invMap_tgt'], mixed_batch['mixed_uniMap_tgt'] = mixed_invMap_tgt, mixed_uniMap_tgt
        
        for k, i in mixed_batch.items():
           
            if 'Map' not in k:
                mixed_batch[k] = torch.cat(i, dim=0)
        
        return mixed_batch
      
def mask(origin_pts, origin_labels, origin_features,
         dest_pts, dest_labels, dest_features,
         sampling_weights=None, is_pseudo=False):
    
    """
    从origin 里面去选 然后放到dest里面
    """
    
    # to avoid when filtered labels are all -1
    if (origin_labels == 0).sum() < origin_labels.shape[0]:
        origin_present_classes = torch.unique(origin_labels)
        origin_present_classes = origin_present_classes[origin_present_classes != 0]
        selection_perc = 0.5
        num_classes = int(selection_perc * origin_present_classes.shape[0])

        selected_classes = sample_classes(origin_present_classes.cpu().numpy(), num_classes, sampling_weights, is_pseudo)

        selected_idx = []
        selected_pts = []
        selected_labels = []
        selected_features = []
        augment_mask_data = True  

        if not augment_mask_data:
            for sc in selected_classes:
                class_idx = torch.where(origin_labels == sc)[0]

                selected_idx.append(class_idx)
                selected_pts.append(origin_pts[class_idx])
                selected_labels.append(origin_labels[class_idx])
                selected_features.append(origin_features[class_idx])

            if len(selected_pts) > 0:
                # selected_idx = np.concatenate(selected_idx, axis=0)
                selected_pts = torch.cat(selected_pts, dim=0) # shape:(9912, 3)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_features = torch.cat(selected_features, dim=0)

        else:
            # 从选到的每个类别里面 取出一半的点 并且进行旋转
            for sc in selected_classes:
                class_idx = torch.where(origin_labels == sc)[0]

                class_pts = origin_pts[class_idx]
                num_pts = class_pts.shape[0]
                sub_num = int(0.5 * num_pts)

                # random subsample
                random_idx = random_sample(class_pts, sub_num=sub_num)
                class_idx = class_idx[random_idx]
                class_pts = class_pts[random_idx]

                # get transformation
                voxel_mtx, affine_mtx = get_transformation_matrix()

                rigid_transformation = affine_mtx @ voxel_mtx
                rigid_transformation = torch.from_numpy(rigid_transformation).float().cuda()
                # apply transformations
                homo_coords = torch.hstack((class_pts, torch.ones((class_pts.shape[0], 1), dtype=class_pts.dtype).cuda()))
                # homo_coords = np.hstack((class_pts, np.ones((class_pts.shape[0], 1), dtype=class_pts.dtype)))
                class_pts = homo_coords @ rigid_transformation.T[:, :3]
                class_labels = torch.ones_like(origin_labels[class_idx]) * sc
                class_features = origin_features[class_idx]

                selected_idx.append(class_idx)
                selected_pts.append(class_pts)
                selected_labels.append(class_labels)
                selected_features.append(class_features)

            if len(selected_pts) > 0:
                # selected_idx = np.concatenate(selected_idx, axis=0)
                selected_pts = torch.cat(selected_pts, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_features = torch.cat(selected_features, dim=0)

        if len(selected_pts) > 0:
            dest_idx = dest_pts.shape[0]
            dest_pts = torch.cat([dest_pts, selected_pts], dim=0)
            dest_labels = torch.cat([dest_labels, selected_labels], dim=0)
            dest_features = torch.cat([dest_features, selected_features], dim=0)

            mask = torch.ones(dest_pts.shape[0])
            mask[:dest_idx] = 0

        augment_data = True    
        if augment_data:
            # get transformation
            voxel_mtx, affine_mtx = get_transformation_matrix()
            rigid_transformation = affine_mtx @ voxel_mtx
            rigid_transformation = torch.from_numpy(rigid_transformation).float().cuda()

            # apply transformations
            homo_coords = torch.hstack((dest_pts, torch.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype).cuda()))

            # homo_coords = np.hstack((dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype)))
            dest_pts = homo_coords @ rigid_transformation.T[:, :3]

    return dest_pts, dest_labels, dest_features, mask.bool()



def sample_classes(origin_classes, num_classes, sampling_weights=None, is_pseudo=False):
    
    weighted_sampling = True   

    if not is_pseudo:
        if weighted_sampling and sampling_weights is not None:

            sampling_weights = sampling_weights[origin_classes] * (1/sampling_weights[origin_classes].sum())

            selected_classes = np.random.choice(origin_classes, num_classes,
                                                replace=False, p=sampling_weights)

        else:
            selected_classes = np.random.choice(origin_classes, num_classes, replace=False)

    else:
        selected_classes = origin_classes

    return selected_classes


def random_sample(points, sub_num):
    """
    :param points: input points of shape [N, 3]
    :return: np.ndarray of N' points sampled from input points
    """

    num_points = points.shape[0]

    if sub_num is not None:
        if sub_num <= num_points:
            sampled_idx = np.random.choice(np.arange(num_points), sub_num, replace=False)
        else:
            over_idx = np.random.choice(np.arange(num_points), sub_num - num_points, replace=False)
            sampled_idx = np.concatenate([np.arange(num_points), over_idx])
    else:
        sampled_idx = np.arange(num_points)

    return sampled_idx

def get_transformation_matrix():

    use_augmentation = True   
    scale_augmentation_bound = (0.95, 1.05)
    rotation_augmentation_bound = (
                                   (-0.15707963267948966, 0.15707963267948966),
                                   (-0.15707963267948966, 0.15707963267948966),
                                   (-0.15707963267948966, 0.15707963267948966)
                                   )
    translation_augmentation_ratio_bound = None

    voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)

    # Transform pointcloud coordinate to voxel coordinate.
    # 1. Random rotation
    rot_mat = np.eye(3)
    if use_augmentation and rotation_augmentation_bound is not None:
        if isinstance(rotation_augmentation_bound, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(rotation_augmentation_bound):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(*rot_bound)
                rot_mats.append(M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
        else:
            raise ValueError()
    rotation_matrix[:3, :3] = rot_mat
    # 2. Scale and translate to the voxel space.
    scale = 1
    if use_augmentation and scale_augmentation_bound is not None:
        scale *= np.random.uniform(*scale_augmentation_bound)
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)

    # 3. Translate
    if use_augmentation and translation_augmentation_ratio_bound is not None:
        tr = [np.random.uniform(*t) for t in translation_augmentation_ratio_bound]
        rotation_matrix[:3, 3] = tr
    # Get final transformation matrix.
    return voxelization_matrix, rotation_matrix


def construct_sparse_tensor(pts, lbls, feats, domain_masks,
                            voxel_size, batch_ind,
                            new_batch, mix_num):
  
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
    
    