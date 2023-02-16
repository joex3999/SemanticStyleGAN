import pandas as pd
import numpy as np
import torch
import importlib.util
import os
import time
from imageio import imread
from scipy import linalg
import argparse
import sys
from loguru import logger
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms as transforms

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from visualize.utils import generate

counter = 0
## FID Function and Classes ##
fid_calc = FrechetInceptionDistance(feature=64)

# def calculate_fid(real_data, generated_data):
#     """Calculates the FSD of two paths"""
#     fid = FrechetInceptionDistance(feature=64)

#     fid.update(real_data, real=True)
#     fid.update(generated_data, real=False)
#     fid = fid.compute()
#     return fid


def initalize_model(ckpt, device):
    ckpt = torch.load(ckpt)
    model = make_model(ckpt["args"])
    model.to(device)
    model.eval()
    model.load_state_dict(ckpt["g_ema"])
    return model


def calculate_mean_for_one_hot(image):
    sem_seg_tensor = torch.tensor(image)
    sem_seg_unsqueezed = sem_seg_tensor.reshape(-1)
    res = torch.nn.functional.one_hot(sem_seg_unsqueezed.to(torch.int64), 16)
    final_res = res.reshape(
        sem_seg_tensor.shape[0], sem_seg_tensor.shape[1], -1
    ).float()
    mean_val = torch.mean(final_res, dim=(0, 1))
    mean_val = mean_val.unsqueeze(0)
    return mean_val


def get_real_images(dataset_path, sample_num=None, batch_size=8):
    dataset = []
    accum = 0
    global fid_calc
    files_count = (
        sum([len(files) for r, d, files in os.walk(dataset_path)])
        if sample_num is None
        else sample_num
    )
    for suffix in ["train", "train_extra", "val"]:
        logger.info(f"Checking dataset {suffix}")
        for subdir, _, files in os.walk(f"{dataset_path}/{suffix}"):
            for file in files:
                if not "leftImg8bit" in file:
                    continue
                if len(dataset) >= sample_num:
                    break
                if accum >= sample_num:
                    return
                accum += 1
                if accum % 10 == 0:
                    logger.info(f"Done with {(accum)} files of the real data")

                filepath = subdir + os.sep + file
                image = imread(filepath)
                image = torch.tensor(image)
                image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)
                dataset.append(image)
                if len(dataset) >= batch_size:
                    dataset = torch.tensor(np.concatenate(dataset))
                    transform = T.Resize((128, 256))
                    dataset = transform(dataset)
                    logger.info("Update real")
                    fid_calc.update(dataset, real=True)
                    dataset = []

    return dataset


def get_data(img_path, batch_size, max_image_num):
    img_list = []
    global counter
    global fid_calc
    for _ in range(batch_size):
        logger.info(counter)
        img = Image.open(f"{img_path}/image_{counter}.png").convert("RGB")
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img = img.unsqueeze(0)
        img_list.append(img)
        counter += 1
        if len(img_list) >= batch_size or counter >= max_image_num:
            imgs = torch.cat(img_list, 0) * 255
            transform = T.Resize((256, 256))
            imgs = transform(imgs)
            logger.info(f"Update fake ")
            fid_calc.update(imgs.to(torch.uint8), real=False)
            return imgs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, help="path to the model checkpoint")
    parser.add_argument("--dataset", type=str, default=8, help="path for dataset")
    parser.add_argument("--batch", type=int, default=8, help="batch size for inference")
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="number of samples to be generated to calculate FSD",
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation ratio"
    )
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=10000,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="running device for inference"
    )
    parser.add_argument(
        "--real_dataset_values",
        type=str,
        default=None,
        help="saved value for real dataset",
    )
    args = parser.parse_args()
    start_time = time.time()
    logger.add("./log_files/logguru/logging_{time}_kid.log")
    logger.info("Calculating!!!")
    ## Settings
    batch_size = 16
    max_img_num = min(args.sample, 3000)
    img_path = "/no_backups/g013/other_GANs/cityscapes_imgs"
    ###

    logger.info(f"Calculating values for the real data.")
    real_images = get_real_images(
        args.dataset, sample_num=args.sample, batch_size=batch_size
    )

    logger.info(
        f"time taken to calculate statistics of real data: {time.time()-start_time}"
    )

    logger.info("Calculating generated cond")

    batch_range = int(max_img_num / batch_size)
    for i in range(batch_range):
        generated_images = get_data(img_path, batch_size, max_img_num)
        logger.info(generated_images.shape)

    logger.info(
        f"time taken to calculate statistics of fake data: {time.time()-start_time}"
    )

    fid = fid_calc.compute()

    logger.info(f" FID Value reported is {fid}")
