import os
import PIL
import numpy as np
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import ListedColormap

import torch
from torchvision import transforms


# def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
#     index = (sem_index_real * num_semantics + sem_index_pred).long().flatten()
#     count = torch.bincount(index)
#     confusion_matrix = torch.zeros((num_semantics, num_semantics)).to(sem_index_real.get_device())
#     for i_label in range(num_semantics):
#         for j_pred_label in range(num_semantics):
#             cur_index = i_label * num_semantics + j_pred_label
#             if cur_index < len(count):
#                 confusion_matrix[i_label, j_pred_label] = count[cur_index]
#     return confusion_matrix

# def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
#     confusion_matrix = None
#     for real_id, pred_id in zip(sem_index_real, sem_index_pred):
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
    # bincount is much faster on cpu than gpu
    device = sem_index_real.device
    sem_index_real = sem_index_real.cpu()
    sem_index_pred = sem_index_pred.cpu()
    mask = (sem_index_real >= 0) & (sem_index_real < num_semantics)
    index = (
        (sem_index_real[mask] * num_semantics + sem_index_pred[mask]).long().flatten()
    )
    confusion_matrix = torch.bincount(index, minlength=num_semantics**2).view(
        num_semantics, num_semantics
    )
    return confusion_matrix.to(device)
