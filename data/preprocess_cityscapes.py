import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from imageio import imread, imwrite
from multiprocessing import Pool
import cv2
from pathlib import Path

# seg_dataset_path = os.path.join(dataset_path, 'CelebAMask-HQ-mask-anno/')
# seg_trainset_path = os.path.join(dataset_path, 'label_train')
# img_valset_path = os.path.join(dataset_path, 'image_val')
# seg_valset_path = os.path.join(dataset_path, 'label_val')

dataset_path = sys.argv[1]  #TODO use this variable
output_prefix = sys.argv[2]

colored_label = False
validation_cutoff=28000

'''
0: unlabled out of roi
1: Flat : road sidewalk parking rail track
2: human : person rider
3: vehicle : car truck bus on rails motorcycle bicycle caravan trailer
4: construction : building wall fence guard rail bridge tunnel
5: object : pole pole group rtaffic sign traffic light
6: nature : vegetation terrain
7: sky : sky
8:void : ground dynamic static
'''
#TODO : unlabeled ? ego vehicle ? rectifiacation border ? out of roi > and lp = -1 ?
cut_down_mapping = {
    0: 0,
    1: 0,
    2: 5,
    3: 0,
    4: 8,
    5: 8,
    6: 8,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 4,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 4,
    17: 5,
    18: 5,
    19: 5,
    20: 5,
    21: 6,
    22: 6,
    23: 7,
    24: 2,
    25: 2,
    26: 3,
    27: 3,
    28: 3,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 3,
}
labels = {
    'unlabeled':  0 ,
    'ego vehicle':  1 ,
    'rectification border':  2 ,
    'out of roi' :  3 ,
    'static' :  4 ,
    'dynamic' :  5 ,
    'ground'    :  6 ,
    'road'      :  7 ,
    'sidewalk'  :  8 ,
    'parking'   :  9 ,
    'rail track': 10 ,
    'building'  : 11 ,
    'wall'      : 12 ,
    'fence'     : 13 ,
    'guard rail': 14 ,
    'bridge'    : 15 ,
    'tunnel'    : 16 ,
    'pole'      : 17 ,
    'polegroup' : 18 ,
    'traffic light': 19 ,
    'traffic sign'  : 20 ,
    'vegetation'    : 21 ,
    'terrain'       : 22 ,
    'sky'           : 23 ,
    'person'        : 24 ,
    'rider'         : 25 ,
    'car'           : 26 ,
    'truck'         : 27 ,
    'bus'           : 28 ,
    'caravan'       : 29 ,
    'trailer'       : 30 ,
    'train'         : 31 ,
    'motorcycle'    : 32 ,
    'bicycle'       : 33 ,
    'license plate' : -1 
}


color_map = {
    0: [0, 0, 0],
    1: [239, 234, 90],
    2: [44, 105, 154],
    3: [4, 139, 168],
    4: [13, 179, 158],
    5: [131, 227, 119],
    6: [185, 231, 105],
    7: [107, 137, 198],
    8: [241, 196, 83],
    9: [242, 158, 76],
    10: [234, 114, 71],
    11: [215, 95, 155],
    12: [207, 113, 192],
    13: [159, 89, 165],
    14: [142, 82, 172],
    15: [158, 115, 200], 
    16: [116, 95, 159],
}

#Reading an image, Simplifing it's labels to only 8 labels instead of 33 
def simplify_image_labels(image,viewable=False):
  new_image = np.zeros(image.shape)
  for k,v in cut_down_mapping.items():
    mask = image==k
    new_image[mask]= (255/8)*v if viewable else v
  return new_image

def process_img():
    accum=0
    for subdir, dirs, files in os.walk(dataset_path):
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
            preprocessed_image=simplify_image_labels(image,True)
            cv2.imwrite(output_path,preprocessed_image)


if __name__ == "__main__":

    # assert os.path.isdir(seg_dataset_path)
    # os.mkdir(seg_trainset_path)
    # os.mkdir(img_valset_path)
    # os.mkdir(seg_valset_path)

    # pool = Pool(16)
    # with tqdm(total=30000) as pbar:
    #     for _ in pool.imap(process_img, range(30000)):
    #         pbar.update()
    process_img()
