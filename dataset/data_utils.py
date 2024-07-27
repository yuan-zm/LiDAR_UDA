import numpy as np
from utils.data_process import DataProcessing as DP
from pathlib import Path
from os.path import join
import numpy as np
import pickle
import os
from sklearn.neighbors import KDTree

def get_sk_data(pc_name, dataset_path, remap_lut, data_name):

    seq_id, frame_id = pc_name[0], pc_name[1]

    point_path = join(dataset_path, seq_id, 'velodyne', frame_id + '.bin')
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.label')
    
    scan = np.fromfile(point_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # put in attribute
    points = scan[:, 0:3]  # get xyz
    remissions = scan[:, 3]  # get remission

    # load labels
    label = np.fromfile(label_path, dtype=np.int32)
    label = label.reshape((-1))
    
    if data_name == 'SemanticKITTI' or data_name == 'SemanticPOSS':
        label = label & 0xFFFF  # semantic label in lower half
    if remap_lut is not None:
        label = remap_lut[label]

    return points, remissions, label

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def augment_scale(points):  # from xmuda
    s_min = 0.95
    s_max = 1.05
    s = (s_max - s_min) * np.random.random() + s_min
    # rot_matrix = np.eye(3, dtype=np.float32)
    # theta = np.random.rand() * rot_z
    # z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                             [np.sin(theta), np.cos(theta), 0],
    #                             [0, 0, 1]], dtype=np.float32)
    # rot_matrix = rot_matrix.dot(z_rot_matrix)
    points = points * s

    return points

def augment_noisy_rot(points, noisy_rot=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix += np.random.randn(3, 3) * noisy_rot
    points = points.dot(rot_matrix)

    return points

def augment_flip_x(points, flip_x=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
    points = points.dot(rot_matrix)

    return points

def augment_flip_y(points, flip_y=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
    points = points.dot(rot_matrix)

    return points

def augment_rot_z(points, rot_z=0.0):  # from xmuda
    rot_matrix = np.eye(3, dtype=np.float32)
    theta = np.random.rand() * rot_z
    z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]], dtype=np.float32)
    rot_matrix = rot_matrix.dot(z_rot_matrix)
    points = points.dot(rot_matrix)

    return points
