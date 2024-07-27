
# Common
import sys
import os
import argparse

import yaml

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.utils.data as torch_data
from torch.utils.data import DataLoader

from tqdm import tqdm

import numpy as np
from utils.data_process import DataProcessing as DP

from dataset.data_utils import get_sk_data

import MinkowskiEngine as ME

# config file
from configs.config_base import cfg_from_yaml_file
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='~/dataset/SynLiDAR/sub_dataset',
        help='Dataset dir. No Default',
    )
parser.add_argument(
        '--sequences',  # '-l',
        nargs="+",
        default= ['00', '01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'] ,
        help='evaluated sequences',
    )
parser.add_argument(
        '--data-name', 
        type=str,
        required=False,
        default="SynLiDAR",
        help='The name of dataset. Default is %(default)s',
    )

parser.add_argument(
        '--tgt-data-name', 
        type=str,
        required=False,
        default="SemanticKITTI",
        help='The name of the target domain dataset. Default is %(default)s',
    )

parser.add_argument(
        '--data-config-file', 
        type=str,
        required=False,
        default="dataset/configs/SynLiDAR2SemanticKITTI/annotations.yaml",
        help='Detail class mapping file.',
    )

parser.add_argument(
        '--class-mapping', 
        type=str,
        required=False,
        default="map_2_semantickitti",
        help='Class mapping name. Default is %(default)s',
    )

parser.add_argument(
        '--num-classes', 
        type=int,
        required=False,
        default=20,
        help='The total number of classes. Default is %(default)s',
    )

parser.add_argument(
        '--voxel-size', 
        type=float,
        required=False,
        default=0.05, # 5cm
        help='Voxel size of voxilization. Default is 5cm',
    )
FLAGS = parser.parse_args()

class mini_dataset(torch_data.Dataset):
    def __init__(self, cfg):
        self.dataset_path = os.path.expanduser(cfg.data_path)
        seq_list = cfg.sequences
        self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        self.data_list = sorted(self.data_list)
        print('This is ** {} ** dataset, filepath is ** {} ** \n \
               voxel size is ** {} **, has ** {} ** scans.'.
               format(cfg.data_name, self.dataset_path, cfg.voxel_size, len(self.data_list)))

        # load data mapping file
        DATA = yaml.safe_load(open(cfg.data_config_file, 'r'))
        if cfg.class_mapping is not None:
            remap_dict = DATA[cfg.class_mapping]
            max_key = max(remap_dict.keys())
            self.remap_lut = np.zeros((max_key + 100), dtype=np.int32)
            self.remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        else:
            self.remap_lut = None
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        pc_name = self.data_list[item]
        pc, remis, lab = get_sk_data(
                                pc_name,
                                self.dataset_path,
                                self.remap_lut,
                                FLAGS.data_name
                                )
        _, v_xyz, v_lab = ME.utils.sparse_quantize(
                                            coordinates=pc,
                                            features=pc ,
                                            labels=lab,
                                            quantization_size=FLAGS.voxel_size
                                            )
        v_lab[v_lab == -100] = 0
        return v_lab
    
    def collate_fn(self, batch):
        v_lbl = []
        for i in range(len(batch)):
            v_lbl.append(batch[i])
        v_lbl_batch = np.hstack(v_lbl)
        
        return v_lbl_batch

cal_dataset = mini_dataset(FLAGS)
cal_dataloader = DataLoader(
                        cal_dataset,
                        batch_size=16,
                        num_workers=4,
                        collate_fn=cal_dataset.collate_fn,
                        shuffle=False,
                        drop_last=False,
                        )

tqdm_cal_dataloader = tqdm(cal_dataloader, total=len(cal_dataloader), ncols=50)

num_per_class = [0 for _ in range(FLAGS.num_classes)]
for  batch_idx, batch_v_label in enumerate(tqdm_cal_dataloader):
   
    inds, counts = np.unique(batch_v_label, return_counts=True)
    for i, c in zip(inds, counts):
        # if i == 0:      # 0 : unlabeled
        #     continue
        # else:
        num_per_class[i] += c
        
print(num_per_class)

num_per_class = np.array(num_per_class)
# 0 is ignore_class. We set it to 0.
num_per_class[0] = 0.
tot = num_per_class.sum()
sampling_weights = 1 - num_per_class / tot
sampling_weights[0] = 0.
print('From {} to {} sampling_weights: \n '.format(FLAGS.data_name, FLAGS.tgt_data_name), sampling_weights.tolist())
# print((num_per_class).tolist())

print('done')



