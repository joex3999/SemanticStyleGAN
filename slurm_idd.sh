#!/bin/bash -l
#SBATCH --job-name="SemanticStyleGan_IDD"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=4-23:00:00
#SBATCH --mail-type=ALL
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=./log_files/IDD_rectangle/SSG%j.%N_output.out
#SBATCH --error=./log_files/IDD_rectangle/SSG%j.%N_error.out
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.3



#Preprocess IDD
#python3.9 ~/SemanticStyleGAN/data/preprocess_idd.py --data="/data/public/idd-segmentation/IDD_Segmentation/gtFine" --output="/no_backups/g013/data/IDD/preprocessed/v5/"

#Prepare image Data 
python3.9 prepare_image_data.py  /data/public/idd-segmentation/IDD_Segmentation/leftImg8bit  --out /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v1_images_only --size 256


#Prepare mask Data 
#python3.9 prepare_mask_data.py --IDD "True" /data/public/idd-segmentation/IDD_Segmentation/leftImg8bit /no_backups/g013/data/IDD/preprocessed/v3 --out /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v3_rectangle --size_h 128 --size_w 256

#Training Inception Network
#python3.9 prepare_inception.py /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v3_rectangle --output /no_backups/g013/data/IDD/inception_models/inception_v3_rectangle.pkl --size_h 128 --size_w 256 --dataset_type mask



#Training using Multiple GPUs
#CUDA_VISIBLE_DEVICES=0,1,2 python3.9 -m torch.distributed.launch --nproc_per_node=3 train.py --dataset /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v2 --inception /no_backups/g013/data/IDD/inception_models/inception_v2.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/IDD_v3  --seg_dim 16 --size 256  --residual_refine 
#13 LG
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v3 --inception /no_backups/g013/data/IDD/inception_models/inception_v3.pkl --save_every 5000   --checkpoint_dir /no_backups/g013/checkpoints/IDD_v9  --seg_dim 13 --size 256  --residual_refine --batch=8

#21LG
#CUDA_VISIBLE_DEVICES=0,1,2 python3.9 -m torch.distributed.launch --nproc_per_node=3 train.py --dataset /no_backups/g013/data/IDD/lmdb_datasets/lmdb_v4 --inception /no_backups/g013/data/IDD/inception_models/inception_v4.pkl --save_every 5000  --checkpoint_dir /no_backups/g013/checkpoints/IDD_v6  --seg_dim 21 --size 256  --residual_refine --ckpt "/no_backups/g013/checkpoints/IDD_v6/ckpt/005000.pt"




#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.10  --seg_dim 16 --size 256  --residual_refine
#Start Training From the Beginning
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.5_2  --seg_dim 16 --size 256  --residual_refine
#CUDA_VISIBLE_DEVICES=0,1,2 python3.9 -m torch.distributed.launch --nproc_per_node=3 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v4.1  --seg_dim 16 --size 256  --residual_refine
#Preprocess CityScapes
#python3.9 ~/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/no_backups/g013/data/gtFine" --output="/no_backups/g013/data/preprocessed/v3.3/"
#Load From Checkpoints
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --ckpt /no_backups/g013/checkpoints/SSG_v3.3/ckpt/200000.pt --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.3 --seg_dim 9 --size 256 --transparent_dims 3 --residual_refine 
#Prepare Data for 5k images
#python3.9 prepare_mask_data.py --cityscapes "True" /no_backups/g013/data/leftImg8bit /no_backups/g013/data/preprocessed/v3.3 --out /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --size 256
##Calculate FSD:
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --dataset="/no_backups/g013/data/preprocessed/v3.3/gtFine_preprocessed" --real_dataset_values="./real_dataset_cond_old.npy" --save_real_dataset "True" --sample=50000
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/SSG_v4.1/ckpt/215000.pt" --dataset="/no_backups/g013/data/preprocessed/v3.3/gtFine_preprocessed" --real_dataset_values="./real_dataset_cond_old.npy" --save_real_dataset "True" --sample=50000
