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

# color_map = {
#     0: [0, 0, 0],
#     1: [239, 234, 90],
#     2: [44, 105, 154],
#     3: [4, 139, 168],
#     4: [13, 179, 158],
#     5: [131, 227, 119],
#     6: [185, 231, 105],
#     7: [107, 137, 198],
#     8: [241, 196, 83],
#     9: [242, 158, 76],
#     10: [234, 114, 71],
#     11: [215, 95, 155],
#     12: [207, 113, 192],
#     13: [159, 89, 165],
#     14: [142, 82, 172],
#     15: [158, 115, 200], 
#     16: [116, 95, 159],
# }
#City Scapes Color Map for version_3
#
color_map = {
    0: [0, 0, 0], #Void
    1: [128, 64,128], #Road
    2: [244, 35,232], # Side Walk
    3: [70, 70, 70], # Building
    4: [102,102,156], # Wall
    5: [190,153,153],#Fence
    6: [153,153,153],# pole
    7: [250,170, 30], # traffic light
    8: [220,220,  0], #Traffic sign
    9: [107,142, 35], # Vegitation
    10: [70,130,180], #sky
    11: [220, 20, 60], #human
    12: [255,  0,  0],#rider
    13: [ 0,  0,142],#car
    14: [ 0, 60,100],# other vehicles
    15: [ 0,  0,230],  #bike and motorcycle
    16: [116, 95, 159],
}

#City Scapes Color Map for version 1 
#
# color_map = {
#     0: [0, 0, 0], #Void
#     1: [128, 64,128], #Road
#     2: [220, 20, 60], # Human
#     3: [ 0,  0,142], # Vehicle
#     4: [70, 70, 70], # Buildings/Construction
#     5: [190,153,153],#Objects
#     6: [107,142, 35],# Naturer
#     7: [70,130,180], #Sky
# }
def generate_img(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images = []
    for head in range(0, styles.size(0), batch_size):
        images_, _ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_)
    images = torch.cat(images,0)
    return tensor2image(images)

def generate(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs)

def generate_tensors(
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
            **kwargs
        )
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images, 0), torch.cat(segs, 0)
    return tensor2image(images), tensor2seg(segs),images, segs

def tensor2image(tensor):
    images = tensor.cpu().clamp(-1,1).permute(0,2,3,1).numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images
    
def tensor2seg(sample_seg):
    seg_dim = sample_seg.size(1)
    sample_seg = torch.argmax(sample_seg, dim=1).detach().cpu().numpy()
    sample_mask = np.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=np.uint8)
    for key in range(seg_dim):
        sample_mask[sample_seg==key] = color_map[key]
    return sample_mask

def cubic_spline_interpolate(styles, step):
    device = styles.device
    styles = styles.detach().cpu().numpy()
    N, K, D = styles.shape
    x = np.linspace(0.0, 1.0, N)
    y = styles.reshape(N,K*D)
    spl = CubicSpline(x, y)
    x_out = np.linspace(0.0, 1.0, step)
    results = spl(x_out) # Step x KD
    results = results.reshape(step,K,D)
    return torch.tensor(results, device=device).float()