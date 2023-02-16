import pandas as pd
import numpy as np
import torch
import os
import time
from imageio import imread
import sys
from loguru import logger
from tqdm import tqdm
import torchvision.transforms as T
import math

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from models.segmentor.models.segmentor import Segmentor
from utils.dataset import MaskDataset, MultiResolutionDataset
from torch.utils import data
from utils.options import Options
from utils.confusion_matrix import get_confusion_matrix
from visualize.utils import generate


##TODO: Remove and import from another place
color_map = {
    0: [0, 0, 0],  # Void
    1: [128, 64, 128],  # Road
    2: [244, 35, 232],  # Side Walk
    3: [70, 70, 70],  # Building
    4: [102, 102, 156],  # Wall
    5: [190, 153, 153],  # Fence
    6: [153, 153, 153],  # pole
    7: [250, 170, 30],  # traffic light
    8: [220, 220, 0],  # Traffic sign
    9: [107, 142, 35],  # Vegitation
    10: [70, 130, 180],  # sky
    11: [220, 20, 60],  # human
    12: [255, 0, 0],  # rider
    13: [0, 0, 142],  # car
    14: [0, 60, 100],  # other vehicles
    15: [0, 0, 230],  # bike and motorcycle
    16: [116, 95, 159],
}
cut_down_mapping_v2 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 1,
    10: 1,
    11: 3,
    12: 4,
    13: 5,
    14: 4,
    15: 3,
    16: 3,
    17: 6,
    18: 6,
    19: 7,
    20: 8,
    21: 9,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 14,
    29: 14,
    30: 14,
    31: 14,
    32: 15,
    33: 15,
}


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def simplify_image_labels(image, viewable=False):
    new_image = np.zeros(image.shape)
    for k, v in cut_down_mapping_v2.items():
        mask = (image == k).cpu()
        new_image[mask] = v if not viewable else ((v) * 255) / 15
    return new_image


def initialize_segmentor(opt):
    ## TODO: Check if distributed
    logger.info("Initalizing Segmentor")
    segmentor = Segmentor(
        opt,
        is_train=False,
        is_main=True,
        logger=logger,
        distributed=False,
    )
    logger.info(f"Segmentor initalized successfuly.")
    # segmentor = engine.data_parallel(segmentor_on_one_gpu)  ##If Distributed
    segmentor.eval()
    return segmentor


def initalize_model(ckpt, device):
    ckpt = torch.load(ckpt)
    model = make_model(ckpt["args"])
    model.to(device)
    model.eval()
    model.load_state_dict(ckpt["g_ema"])
    return model


def prepare_data_sampler(lmdb_path, batch_size, device="cuda"):
    dataset = MaskDataset(lmdb_path, resolution=256, label_size=16)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        num_workers=4 // 2,
        drop_last=True,
    )

    return loader


def compute_confusion_matrix(opt, data, model, num_semantics=17):
    with torch.no_grad():
        # data["img"] = (
        #     data["img"] * 2 / 255
        # ) - 1  # convert image range from [-1 to 1]
        pred_seg = model(data, mode="inference", hard=False)
    pred_sem_seg = pred_seg["sem_seg"]

    sem_index_pred = pred_sem_seg.max(dim=1, keepdim=True)[1]
    #No Need to simplify image labels in the case of a model trained on 16 classes.
    #sem_index_pred = torch.from_numpy(simplify_image_labels(sem_index_pred, False))
    sem_index_real = data["sem_seg"].cuda()
    sem_index_real = sem_index_real.unsqueeze(1)

    ##TODO: Calculate if num of semantics is 16,15, and once if it is 35 for example.
    logger.info(
        f"sem_index_real shape :{sem_index_real.shape} sem_index_pred shape:{sem_index_pred.shape}"
    )
    logger.info(
        f"sem_index_real max :{sem_index_real.max()} sem_index_pred max:{sem_index_pred.max()}"
    )
    confusion_matrix = get_confusion_matrix(
        sem_index_real, sem_index_pred, num_semantics=num_semantics
    )
    logger.info("Calculated confusion matrix. Successfuly")
    return confusion_matrix.cpu()


def compute_iou(eval_idx, confusion_matrix):
    pos = confusion_matrix.sum(dim=1)
    res = confusion_matrix.sum(dim=0)
    tp = torch.diag(confusion_matrix)
    iou = tp / torch.max(torch.Tensor([1.0]), pos + res - tp)
    mean_iou = iou.mean()
    pos_eval = pos[eval_idx]
    res_eval = confusion_matrix[eval_idx].sum(dim=0)[eval_idx]
    tp_eval = tp[eval_idx]
    iou_eval = tp_eval / torch.max(torch.Tensor([1.0]), pos_eval + res_eval - tp_eval)
    mean_iou_eval = iou_eval.mean()
    return mean_iou, mean_iou_eval, iou, iou_eval


if __name__ == "__main__":
    start_time = time.time()
    logger.add("./log_files/logguru/logging_{time}_miou.log")

    opt = Options().parse(
        load_segmentor=True,
        load_seg_generator=True,
        load_img_generator=True,
        load_extra_dataset=True,
        save=True,
    )
    # TODO: Add all to opts

    logger.info("Currently calculating train MIOU")
    segmentor_opt = opt["segmentor"]
    batch_size = 16
    seg_batch_size = 16
    device = "cuda"
    eval_idx = range(1, 16, 1)

    if not segmentor_opt.lmdb:
        print("LMDB path for the dataset should be provided")

    segmentor = initialize_segmentor(segmentor_opt)
    main_loader = prepare_data_sampler(
        segmentor_opt.lmdb,
        batch_size,
        device,
    )
    print(f"length of the dataset is {len(main_loader.dataset)}")
    loader = sample_data(main_loader)

    batch_range = int(len(main_loader.dataset) / seg_batch_size)

    gan_train_confusion_matrix = torch.zeros(
        (segmentor_opt.res_semantics, segmentor_opt.res_semantics)
    )
    # Segmenter takes input as batch size 16
    for i in range(0, batch_range):
        r_data = next(loader)
        images, segmentations = r_data["image"], r_data["mask"]
        images, segmentations = images.to(device), segmentations.to(device)
        segmentations = torch.argmax(segmentations, 1)
        segmentations = segmentations.squeeze(1)
        transform = T.Resize((128, 256))
        images = transform(images)
        segmentations = transform(segmentations)
        data = {
            "img": images.float(),
            "sem_seg": segmentations.float(),
        }

        gan_train_confusion_matrix += compute_confusion_matrix(
            segmentor_opt, data, segmentor, segmentor_opt.res_semantics
        )
    iou = compute_iou(eval_idx, gan_train_confusion_matrix)
    train_mean_iou = iou[0].numpy()
    train_mean_iou_eval = iou[1].numpy()
    train_iou = iou[2].numpy()
    train_iou_eval = iou[3].numpy()

    logger.info(
        f"GAN-train miou values ;\n mean iou :{np.mean(train_mean_iou)} \n mean iou eval:{np.mean(train_mean_iou_eval)}\n , Time Taken in total : {time.time()-start_time}"
    )
