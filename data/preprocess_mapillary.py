import os
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from imageio import imread, imwrite
from multiprocessing import Pool
import cv2
from pathlib import Path

sys.path.insert(0, "../SemanticStyleGAN")
from data import mapillary_mapping


# cut_down_mapping = idd_mapping.cut_down_mapping_v4_level1

cut_down_mapping = mapillary_mapping.cut_down_mapping_v1
# validation_cutoff = 28000

# Reading an image, Simplifing it's labels to only 8 labels instead of 33
def simplify_image_labels(image, viewable=False):
    new_image = np.zeros(image.shape)
    instance_array = np.array(image, dtype=np.uint16)
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)

    for k, v in cut_down_mapping.items():
        mask = instance_label_array == k
        new_image[mask] = (
            (255 / 19) * v if viewable else v
        )  # 19 or max class basically.
    return new_image


def process_img(dataset_path, output_prefix):
    accum = 0

    files_count = sum([len(files) for r, d, files in os.walk(dataset_path)])
    for subdir, _, files in os.walk(dataset_path):
        # The dataset has to be of format */gtFine/train/*/*.png
        # print(subdir.split("/"))
        if subdir.split("/")[-1] == "instances" and subdir.split("/")[-2] == "v1.2":
            print("ok")
            output_dir = Path(output_prefix)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            continue
        for file in files:
            accum += 1
            # print(file)
            if accum % 1000 == 0:
                print(f"Done with {accum}/{files_count} images")

            filepath = subdir + os.sep + file
            output_path = str(output_dir) + os.sep + file
            # print(output_path)
            if not Path(output_path).exists():
                print("does not exists")
            # image = imread(filepath)
            # preprocessed_image = simplify_image_labels(image, False)
            # cv2.imwrite(output_path, preprocessed_image)
        print(f"All files has been processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    process_img(args.dataset, args.output)
