import pandas as pd
import numpy as np
import torch
import os
import time
from imageio import imread, imwrite
from scipy import linalg
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from visualize.utils import generate


# import matplotlib.pyplot as plt
##TODO: Remove and import it from another class
# TODO: Refactor code
## TODO: Save original image statistics to someplace for faster calculation.
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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # msg = ('fid calculation produces singular product; '
        #        'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fsd(real_data, generated_data):
    """Calculates the FSD of two paths"""
    m1 = np.mean(real_data, axis=0)
    s1 = np.cov(real_data, rowvar=False)

    m2 = np.mean(generated_data, axis=0)
    s2 = np.cov(generated_data, rowvar=False)
    fsd_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fsd_value


def initalize_model(ckpt, device):
    ckpt = torch.load(ckpt)
    model = make_model(ckpt["args"])
    model.to(device)
    model.eval()
    model.load_state_dict(ckpt["g_ema"])
    return model


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


def calculate_mean_for_one_hot(image):
    sem_seg_tensor = torch.tensor(image)
    sem_seg_squeezed = sem_seg_tensor.reshape(-1)
    res = torch.nn.functional.one_hot(sem_seg_squeezed.to(torch.int64), -1)
    final_res = res.reshape(
        sem_seg_tensor.shape[0], sem_seg_tensor.shape[1], -1
    ).float()
    mean_val = torch.mean(final_res, dim=(0, 1))
    return mean_val


def calculate_real_cond(dataset_path):
    dataset_mean_values = []
    accum = 0
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            accum += 1
            if accum % 10 == 0:
                print(f"Done with {(accum/5000)*100}% of the real data")
            if "labelIds" not in file:
                continue
            filepath = subdir + os.sep + file
            image = imread(filepath)
            mean_val = calculate_mean_for_one_hot(image)
            dataset_mean_values.append(mean_val.cpu().numpy())
    real_cond = np.concatenate(dataset_mean_values)
    return real_cond


def calculate_generated_cond(ckpt, sample, truncation, truncation_mean, batch, device):
    print(f"Loading model from checkpoint")
    model = initalize_model(ckpt, "cuda")
    print(f"Model initalized successfuly")
    mean_latent = model.style(
        torch.randn(truncation_mean, model.style_dim, device=device)
    ).mean(0)
    start_time = time.time()
    dataset_mean_values = []
    with torch.no_grad():
        n_batch = sample // batch
        resid = sample - (n_batch * batch)
        batch_sizes = [batch] * n_batch + [resid]
        for batch_iter in tqdm(batch_sizes):
            if batch_iter < batch:
                print(f"Skipping batch iteration of size {batch_iter}")
                continue

            styles = model.style(
                torch.randn(batch_iter, model.style_dim, device=device)
            )
            styles = truncation * styles + (1 - truncation) * mean_latent.unsqueeze(0)
            images, segs = generate(
                model, styles, mean_latent=mean_latent, batch_size=batch
            )

            for i in range(len(images)):
                converted_seg = from_rgb_to_label(segs[i], color_map)
                mean_val = calculate_mean_for_one_hot(converted_seg)
                dataset_mean_values.append(mean_val.cpu().numpy())
        print(f"Time taken to generate images : {time.time()-start_time}")
        generated_cond = np.concatenate(dataset_mean_values)
    print(f"Average speed: {(time.time() - start_time)/(sample)}s")
    return generated_cond


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, help="path to the model checkpoint")
    parser.add_argument("--dataset", type=str, default=8, help="path for dataset")
    parser.add_argument("--batch", type=int, default=8, help="batch size for inference")
    parser.add_argument(
        "--sample",
        type=int,
        default=50000,
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
    print("Calculating!!!")
    if args.real_dataset_values:
        print(f"loading file for real dataset values from :{args.real_dataset_values}")
        real_cond = np.load(args.real_dataset_values)
    else:
        print(f"Calculating values for the real data.")
        real_cond = calculate_real_cond(args.dataset)
    print(f"time taken to calculate statistics of real data: {time.time()-start_time}")
    print("Calculating generated cond")
    generated_cond = calculate_generated_cond(
        args.ckpt,
        args.sample,
        args.truncation,
        args.truncation_mean,
        args.batch,
        args.device,
    )
    print(f"time taken to calculate statistics of fake data: {time.time()-start_time}")
    fsd = calculate_fsd(real_cond, generated_cond)
    print(
        f"FSD value by semantic pallete method is {fsd} , Time Taken in total : {time.time()-start_time}"
    )