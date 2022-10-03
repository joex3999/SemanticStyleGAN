
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
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from visualize.utils import generate

## KID Function and Classes ##

def calculate_kid(real_data, generated_data):
    """Calculates the FSD of two paths"""
    kid = KernelInceptionDistance(subset_size=50)

    kid.update(real_data, real=True)
    kid.update(generated_data, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std


def calculate_fid(real_data, generated_data):
    """Calculates the FSD of two paths"""
    fid = FrechetInceptionDistance(feature=64)

    fid.update(real_data, real=True)
    fid.update(generated_data, real=False)
    fid = fid.compute()
    return fid

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
    mean_val=mean_val.unsqueeze(0)
    return mean_val

def get_real_images(dataset_path,sample_num=None):
    dataset = []
    accum = 0
    files_count = sum([len(files) for r, d, files in os.walk(dataset_path)]) if sample_num is None else sample_num
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            accum += 1
            if accum % 10 == 0:
                logger.info(f"Done with {(accum/files_count)*100}% of the real data")
            if not "leftImg8bit" in file:
                continue
            if len(dataset)>=sample_num:
                break
            filepath = subdir + os.sep + file
            image = imread(filepath)
            image = torch.tensor(image)
            image= image.permute(2,0,1)
            image = image.unsqueeze(0)
            dataset.append(image)
    dataset = torch.tensor(np.concatenate(dataset))
    return dataset


def get_generated_images(ckpt, sample, truncation, truncation_mean, batch, device):
    logger.info(f"Loading model from checkpoint")
    model = initalize_model(ckpt, "cuda")
    logger.info(f"Model initalized successfuly")
    mean_latent = model.style(
        torch.randn(truncation_mean, model.style_dim, device=device)
    ).mean(0)
    start_time = time.time()
    dataset = []
    with torch.no_grad():
        n_batch = sample // batch
        resid = sample - (n_batch * batch)
        batch_sizes = [batch] * n_batch + [resid]
        for batch_iter in tqdm(batch_sizes):
            if batch_iter < batch:
                logger.info(f"Skipping batch iteration of size {batch_iter}")
                continue

            styles = model.style(
                torch.randn(batch_iter, model.style_dim, device=device)
            )
            styles = truncation * styles + (1 - truncation) * mean_latent.unsqueeze(0)
            images, segs = generate(
                model, styles, mean_latent=mean_latent, batch_size=batch
            )

            for i in range(len(images)):
                image = torch.tensor(images[i])
                image= image.permute(2,0,1)
                image = image.unsqueeze(0)
                dataset.append(image)
        logger.info(f"Time taken to generate images : {time.time()-start_time}")
        dataset = torch.tensor(np.concatenate(dataset))
    logger.info(f"Average speed: {(time.time() - start_time)/(sample)}s")
    return dataset


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
    logger.add("/usr/stud/faragy/storage/user/logs/logguru/logging_{time}.log")
    logger.info("Calculating!!!")
   
    logger.info(f"Calculating values for the real data.")
    real_images = get_real_images(args.dataset,sample_num=args.sample)
    logger.info(f"time taken to calculate statistics of real data: {time.time()-start_time}")
   
    logger.info("Calculating generated cond")
    generated_images = get_generated_images(
        args.ckpt,
        args.sample,
        args.truncation,
        args.truncation_mean,
        args.batch,
        args.device,
    )
    
    logger.info(f"time taken to calculate statistics of fake data: {time.time()-start_time}")
    kid_mean,kid_std = calculate_kid(real_images, generated_images)
    logger.info(
        f"KID value by semantic pallete method is {kid_mean} with deviation {kid_std} , Time Taken in total : {time.time()-start_time}"
    )
    fid = calculate_fid(real_images,generated_images)

    logger.info(f" FID Value reported is {fid}")