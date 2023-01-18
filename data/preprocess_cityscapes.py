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
import cityscapes_mapping


#cut_down_mapping_v1 = cityscapes_mapping.cut_down_mapping_v1
cut_down_mapping_v2 = cityscapes_mapping.cut_down_mapping_v4

# validation_cutoff = 28000

# Reading an image, Simplifing it's labels to only 8 labels instead of 33
def simplify_image_labels(image, viewable=False):
    new_image = np.zeros(image.shape)
    for k, v in cut_down_mapping_v2.items(): 
        mask = image == k
        new_image[mask] = (
            (255 / 19) * v if viewable else v
        )  # 19 or max class basically.
    return new_image


def process_img(dataset_path, output_prefix):
    accum = 0
    files_count = sum([len(files) for r, d, files in os.walk(dataset_path)])
    for subdir, _, files in os.walk(dataset_path):
        # The dataset has to be of format */gtFine/train/*/*.png
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
            extra_data = "leftImg8bit" in file
            # In normal dataset, label images contain labelIds prefix.
            if not extra_data and "labelIds" not in file:
                continue
            # In extra dataset, files does not contain _prob prefix.
            if extra_data and "_prob" in file:
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
