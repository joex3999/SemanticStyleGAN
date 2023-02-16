# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under The MIT License (MIT)
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse
import torch
from models import make_model
import torchvision.transforms as T
import functools
from utils.inception_utils import sample_gema, prepare_inception_metrics
from PIL import Image
import torchvision.transforms as transforms

counter = 0


def get_data(img_path, batch_size, max_image_num):
    img_list = []
    global counter
    for _ in range(batch_size):
        # For SSG output
        #img = Image.open(f"{img_path}/{counter:06d}_img.png").convert("RGB")
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
            return imgs


def sample_images(img_loc, batch_size, max_image_num, device):
    images = get_data(img_loc, batch_size, max_image_num)
    images = images.to(device)
    # images = ((images / 255) * 4) - 2
    images = ((images / 255) * 2) - 1
    return images


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Calculate FID score for generators",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        help="path to the checkpoint file",
    )
    parser.add_argument(
        "--inception",
        type=str,
        required=True,
        help="pre-calculated inception file",
    )
    parser.add_argument(
        "--batch", default=8, type=int, help="batch size for inception networks"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of samples used for embedding calculation",
    )
    parser.add_argument(
        "--image_directory",
        type=str,
        default=None,
        help="directory that contains images used to calculate FID, This is used to calculate FID of external models",
    )
    parser.add_argument(
        "--max_image_number",
        type=int,
        default=None,
        help="Maximum amount of images present in the directory.",
    )
    args = parser.parse_args()
    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    if args.image_directory is None and args.max_image_number is None:
        print("Loading model...")
        ckpt = torch.load(args.ckpt)
        g_args = ckpt["args"]
        model = make_model(g_args).to(device).eval()
        model.load_state_dict(ckpt["g_ema"])
        mean_latent = model.style(torch.randn(50000, 512, device=device)).mean(0)

        sample_fn = functools.partial(
            sample_gema,
            g_ema=model,
            device=device,
            truncation=1.0,
            mean_latent=None,
            batch_size=args.batch,
        )
    else:
        args.n_sample = min(args.n_sample, args.max_image_number)
        sample_fn = functools.partial(
            sample_images,
            img_loc=args.image_directory,
            batch_size=args.batch,
            max_image_num=args.n_sample,
            device=device,
        )
    print("==================Start calculating FID==================")
    IS_mean, IS_std, FID = get_inception_metrics(
        sample_fn, num_inception_images=args.n_sample, use_torch=False
    )
    print(
        "FID: {0:.4f}, IS_mean: {1:.4f}, IS_std: {2:.4f}".format(FID, IS_mean, IS_std)
    )
