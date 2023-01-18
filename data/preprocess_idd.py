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

sys.path.insert(0, "../../SemanticStyleGAN")
from data import idd_mapping


cut_down_mapping = idd_mapping.cut_down_mapping_v4_level1
# validation_cutoff = 28000

# Reading an image, Simplifing it's labels to only 8 labels instead of 33
def simplify_image_labels(image, viewable=False):
    new_image = np.zeros(image.shape)
    for k, v in cut_down_mapping.items():  ##TODO: Currently using v1
        mask = image == k
        new_image[mask] = (
            (255 / 19) * v if viewable else v
        )  # 19 or max class basically.
    return new_image


def process_img(dataset_path, output_prefix, level_3_ids=False):
    accum = 0
    if level_3_ids:
        print("Converting from Level 3Ids instead of normal ids")
    else:
        print("Converting from normal ID maps")
    files_count = sum([len(files) for r, d, files in os.walk(dataset_path)])
    for subdir, _, files in os.walk(dataset_path):
        # The dataset has to be of format */gtFine/train/*/*.png
        print(subdir.split("/"))
        if subdir.split("/")[-3] == "gtFine":
            output_dir = "/".join(subdir.split("/")[-3:]).replace(
                "gtFine", "gtFine_preprocessed"
            )
            output_dir = Path(output_prefix + output_dir)
            # print(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            accum += 1
            if accum % 1000 == 0:
                print(f"Done with {accum}/{files_count} images")
            if level_3_ids and "labellevel3Ids" not in file:
                continue
            if (not level_3_ids) and "labelids" not in file:
                continue
            filepath = subdir + os.sep + file
            output_path = str(output_dir) + os.sep + file
            image = imread(filepath)
            preprocessed_image = simplify_image_labels(image, False)
            cv2.imwrite(output_path, preprocessed_image)
    print(f"All files has been processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    process_img(args.dataset, args.output)
