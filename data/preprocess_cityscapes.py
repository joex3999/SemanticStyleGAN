import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from imageio import imread, imwrite
from multiprocessing import Pool
import cv2
from pathlib import Path
import cityscapes_mapping

dataset_path = sys.argv[1]  
output_prefix = sys.argv[2]

validation_cutoff=28000
cut_down_mapping_v1=cityscapes_mapping.cut_down_mapping_v1

#Reading an image, Simplifing it's labels to only 8 labels instead of 33 
def simplify_image_labels(image,viewable=False):
  new_image = np.zeros(image.shape)
  for k,v in cut_down_mapping_v1.items():
    mask = image==k
    new_image[mask]= (255/8)*v if viewable else v
  return new_image

def process_img():
    accum=0
    for subdir, _ , files in os.walk(dataset_path):
        if(subdir.split("/")[-3]=="gtFine"):
          output_dir = "/".join(subdir.split("/")[-3:]).replace("gtFine","gtFine_preprocessed")
          output_dir= Path(output_prefix+output_dir)
          #print(output_dir)
          output_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            accum+=1
            if accum%1000 ==0:
                print(f"Done with {(accum/20000)*100}% of the data")
            if "labelIds" not in file:
                continue
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            output_path= str(output_dir) + os.sep + file
            image = imread(filepath)
            preprocessed_image=simplify_image_labels(image,False)
            cv2.imwrite(output_path,preprocessed_image)


if __name__ == "__main__":
    process_img()
