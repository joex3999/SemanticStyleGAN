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
from utils.options import Options
from utils.confusion_matrix import get_confusion_matrix
from visualize.utils import generate
import torchvision.transforms as transforms
from PIL import Image

"""
NOTE: This file is was created to retrieve one metric. 
IT IS EXTREMELY HARD CODED. PLEASE DO NOT CHANGE OR USE.
"""
##TODO: Remove and import from another place
counter = 0
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


def simplify_image_labels(image, viewable=False):
    new_image = np.zeros(image.shape)
    for k, v in cut_down_mapping_v2.items():
        mask = (image == k).cpu()
        new_image[mask] = v if not viewable else ((v) * 255) / 15
    return new_image


# Stupid way of converting from rgb segmentation maps to the original labels
# converting from a label map back to it's label by summing the 3 channels into 1 channel, then
# checking mapping the knows sums to the correct label(index)
def from_rgb_to_label(image, color_map):
    color_map_sum = {}
    new_image = np.zeros((image.shape[0], image.shape[1]))
    for c in color_map:
        color_map_sum[c] = sum(color_map[c])
    image_tensor = torch.tensor(image).to(float)
    image_summed = torch.sum(image_tensor, dim=2)
    for index in color_map_sum:
        mask = image_summed == color_map_sum[index]
        new_image[mask] = index
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


def get_data(img_path, seg_path, batch_size, max_image_num):
    img_list = []
    seg_list = []
    global counter
    for _ in range(batch_size):
        seg = torch.load(
            f"{seg_path}/image_{counter}.pt", map_location=torch.device("cpu")
        )
        img = Image.open(f"{img_path}/image_{counter}.png").convert("RGB")
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img = img.unsqueeze(0)
        seg_list.append(seg)
        img_list.append(img)
        counter += 1
        if len(seg_list) >= batch_size or counter >= max_image_num:
            imgs = torch.cat(img_list, 0) * 255
            segs = torch.cat(seg_list, 0)
            segs = torch.argmax(segs, dim=1)
            transform = T.Resize((128, 256))
            imgs = transform(imgs)
            segs = transform(segs)
            return imgs, segs


def compute_confusion_matrix(opt, data, model, num_semantics=17):
    with torch.no_grad():
        # np.save("./notebooks/data/segmentation_model_input_SSG.npy", data["img"].cpu())
        data["img"] = (data["img"] * 2 / 255) - 1  # convert image range to [-1 to 1]
        pred_seg = model(data, mode="inference", hard=False)
    pred_sem_seg = pred_seg["sem_seg"]

    sem_index_pred = pred_sem_seg.max(dim=1, keepdim=True)[1]
    sem_index_pred = torch.from_numpy(simplify_image_labels(sem_index_pred, False))
    sem_index_real = data["sem_seg"].cuda()
    sem_index_real = torch.from_numpy(simplify_image_labels(sem_index_real, False))
    sem_index_real = sem_index_real.unsqueeze(1)
    logger.info(sem_index_real.shape)
    # np.save("./notebooks/data/segmentation_model_output_SSG.npy", pred_sem_seg.cpu())
    # np.save("./notebooks/data/real_segmentation_input.npy", data["img"].cpu())
    # np.save("./notebooks/data/out_model_output.npy", data["sem_seg"].cpu())

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
    segmentor_opt = opt["segmentor"]
    batch_size = 8
    seg_batch_size = 16
    device = "cuda"
    eval_idx = range(1, 16, 1)

    ## MIOU SPECIFIC PARAMETERS ##
    seg_path = "/no_backups/g013/other_GANs/cityscapes_segs"
    img_path = "/no_backups/g013/other_GANs/cityscapes_imgs"
    max_img_num = 3000
    segmentor = initialize_segmentor(segmentor_opt)
    ######
    logger.info(
        f"time taken to calculate statistics of fake data: {time.time()-start_time}"
    )
    gan_test_confusion_matrix = torch.zeros(
        (segmentor_opt.res_semantics, segmentor_opt.res_semantics)
    )
    # Segmenter takes input as batch size 16
    batch_range = int(max_img_num / batch_size)
    for i in range(batch_range):
        images, segmentations = get_data(img_path, seg_path, batch_size, max_img_num)

        print(
            f"shape of images and segmentations : {images.shape}  /  {segmentations.shape}"
        )
        data = {
            "img": images.float(),
            "sem_seg": segmentations.float(),
        }
        gan_test_confusion_matrix += compute_confusion_matrix(
            segmentor_opt, data, segmentor, segmentor_opt.res_semantics
        )
    iou = compute_iou(eval_idx, gan_test_confusion_matrix)
    test_mean_iou = iou[0].numpy()
    test_mean_iou_eval = iou[1].numpy()
    test_iou = iou[2].numpy()
    test_iou_eval = iou[3].numpy()
    # logger.info(
    #     f"GAN-test miou value calculated is : {np.mean(test_iou)} eval : {np.mean(test_iou_eval)}  mean iou :{np.mean(test_mean_iou)}  mean iou eval:{np.mean(test_mean_iou_eval)} , Time Taken in total : {time.time()-start_time}"
    # )
    logger.info(
        f"GAN-test miou values ;\n mean iou :{np.mean(test_mean_iou)} \n mean iou eval:{np.mean(test_mean_iou_eval)}\n , Time Taken in total : {time.time()-start_time}"
    )
