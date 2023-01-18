# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
import torch
from scipy.interpolate import CubicSpline
import sys

# sys.path.insert(0, "../SemanticStyleGAN")
# from visualize.color_maps import color_map_v6_24C
#TODO: Import color mappings correctly
color_map_v6_24C = {
    0: [0, 0, 0],
    1: [128, 64, 128],
    2: [244, 35, 232],
    3: [220, 20, 60],
    4: [255, 127, 80],
    5: [255, 215, 0],
    6: [0, 0, 142],
    7: [0, 0, 230],
    8: [0, 0, 139],
    9: [0, 0, 200],
    10: [0, 60, 100],
    11: [202, 202, 156],
    12: [102, 102, 156],
    13: [190, 153, 153],
    14: [220, 220, 0],
    15: [250, 170, 30],
    16: [153, 153, 153],
    17: [70, 70, 70],
    18: [101, 101, 101],
    19: [107, 142, 35],
    20: [70, 130, 180],
    21: [128, 128, 128],
    22: [224, 224, 224],
    23: [153, 255, 255],
}

color_map = color_map_v6_24C


def generate_img(
    model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs
):
    images = []
    for head in range(0, styles.size(0), batch_size):
        images_, _ = model(
            [styles[head : head + batch_size]],
            input_is_latent=True,
            truncation=truncation,
            truncation_latent=mean_latent,
            *args,
            **kwargs,
        )
        images.append(images_)
    images = torch.cat(images, 0)
    return tensor2image(images)


def generate(
    model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs
):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model(
            [styles[head : head + batch_size]],
            input_is_latent=True,
            truncation=truncation,
            truncation_latent=mean_latent,
            *args,
            **kwargs,
        )
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images, 0), torch.cat(segs, 0)
    return tensor2image(images), tensor2seg(segs)


def generate_tensor(
    model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs
):  # TODO: combine this and generate into one func
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model(
            [styles[head : head + batch_size]],
            input_is_latent=True,
            truncation=truncation,
            truncation_latent=mean_latent,
            *args,
            **kwargs,
        )
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images, 0), torch.cat(segs, 0)
    return images, segs


def tensor2image(tensor):
    images = tensor.cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def tensor2seg(sample_seg):
    seg_dim = sample_seg.size(1)
    sample_seg = torch.argmax(sample_seg, dim=1).detach().cpu().numpy()
    sample_mask = np.zeros(
        (sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3),
        dtype=np.uint8,
    )
    for key in range(seg_dim):
        sample_mask[sample_seg == key] = color_map[key]
    return sample_mask


def cubic_spline_interpolate(styles, step):
    device = styles.device
    styles = styles.detach().cpu().numpy()
    N, K, D = styles.shape
    x = np.linspace(0.0, 1.0, N)
    y = styles.reshape(N, K * D)
    spl = CubicSpline(x, y)
    x_out = np.linspace(0.0, 1.0, step)
    results = spl(x_out)  # Step x KD
    results = results.reshape(step, K, D)
    return torch.tensor(results, device=device).float()
