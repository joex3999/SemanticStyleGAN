import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import array, cov, mean
from numpy.linalg import eig

sys.path.insert(0, "../SemanticStyleGAN")
from models import make_model
from visualize.utils import cubic_spline_interpolate, generate

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
    16: [116, 95, 159],  # Does not exists
}

latent_dict_cityscapes = {
    2: "void_1",
    3: "void_2",
    4: "road_shape",
    5: "road_texture",
    6: "swalk_shape",
    7: "swalk_texture",
    8: "building_shape",
    9: "building_texture",
    10: "wall_shape",
    11: "wall_texture",
    12: "fence_shape",
    13: "fence_texture",
    14: "pole_shape",
    15: "pole_texture",
    16: "trlight_shape",
    17: "trlight_texture",
    18: "trsign_shape",
    19: "trsign_texture",
    20: "veget_shape",
    21: "veget_texture",
    22: "sky_shape",
    23: "sky_texture",
    24: "person_shape",
    25: "person_texture",
    26: "rider_shape",
    27: "rider_texture",
    28: "car_shape",
    29: "car_texture",
    30: "otherV_shape",
    31: "otherV_texture",
    32: "bike_shape",
    33: "bike_texture",
    0: "coarse_1",
    1: "coarse_2",
}


class Control:

    def __init__(
        self,
        ckpt_dir,
        device,
        batch=8,
        sample=10,
        steps=160,
        truncation=0.7,
        truncation_mean=10000,
    ):
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.batch = batch
        self.sample = sample
        self.steps = steps
        self.truncation = truncation
        self.truncation_mean = truncation_mean
        self.load_model()

    def load_model(self):
        print("Loading model ...")
        ckpt = torch.load(self.ckpt_dir, map_location=torch.device("cpu"))
        self.model = make_model(ckpt["args"])
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(ckpt["g_ema"])
        self.mean_latent = self.model.style(
            torch.randn(self.truncation_mean, self.model.style_dim, device=self.device)
        ).mean(0)

    def from_rgb_to_label(self, image, color_map):
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

    ## Gets class distribution from segmentation map
    def get_class_dist(self, seg, color_map=color_map):
        converted_seg = self.from_rgb_to_label(seg, color_map)
        sem_seg_tensor = torch.tensor(converted_seg)
        sem_seg_unsqueezed = sem_seg_tensor.reshape(-1)
        res = torch.nn.functional.one_hot(sem_seg_unsqueezed.to(torch.int64), 16)
        final_res = res.reshape(
            sem_seg_tensor.shape[0], sem_seg_tensor.shape[1], -1
        ).float()
        mean_val = torch.mean(final_res, dim=(0, 1))
        mean_val = mean_val.unsqueeze(0)
        return mean_val

    def images_to_video(self, images, segs, save_dir):
        frames = [np.concatenate((img, seg), 1) for (img, seg) in zip(images, segs)]
        imageio.mimwrite(f"{save_dir}", frames, fps=20)

    def generate_and_plot_image(
        self, styles, class_index, coords=None, plot=True, get_image=True
    ):
        image, seg = generate(
            self.model,
            styles[0].unsqueeze(0),
            mean_latent=self.mean_latent,
            randomize_noise=False,
            batch_size=self.batch,
            coords=coords,
        )

        all_classes = self.get_class_dist(seg[0], color_map)
        class_percentage = (all_classes[0][class_index] / all_classes.sum()) * 100

        if plot:
            print(f"Class percentage is {class_percentage}")
            plt.imshow(np.concatenate((image[0], seg[0]), 1))
            plt.show()
        if get_image:
            return image, seg
        else:
            return float(class_percentage)

    def edit_image(
        self,
        latent_index,
        class_index,
        change_factor,
        styles,
        addition=True,
        plot=True,
        get_image=True,
        add_mean_latent=False,
    ):

        styles_copy = styles.clone().detach()
        if addition:
            styles_copy[0, latent_index] += change_factor
        else:
            styles_copy[0, latent_index] *= change_factor
        if add_mean_latent:
            styles_copy[0, latent_index] = 0.8 * styles_copy[0, latent_index] + (
                1 - 0.8
            ) * self.mean_latent.unsqueeze(0)
        return self.generate_and_plot_image(
            styles_copy, class_index, plot=plot, get_image=get_image
        )

    def edit_image_principal_component(
        self,
        latent_index,
        class_index,
        change_factor,
        styles,
        principal_component,
        whole_image=False,
        plot=True,
        get_image=True,
    ):

        styles_copy = styles.clone().detach()
        if not whole_image:
            if isinstance(principal_component, list):
                for pc in principal_component:
                    styles_copy[0, latent_index] += pc * change_factor
            else:
                styles_copy[0, latent_index] += principal_component * change_factor
        else:
            principal_component = (
                principal_component.unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, self.model.n_latent, 1)
            )
            styles_copy = styles_copy + (principal_component.float() * change_factor)
        return self.generate_and_plot_image(
            styles_copy, class_index, plot=plot, get_image=get_image
        )
