#!/bin/bash
#SBATCH --job-name="train-faragy"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus=4
#SBATCH --time=4-23:00:00
#SBATCH --mail-type=ALL
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=/usr/stud/faragy/storage/user/logs/SSG_v3.12/%j_output.out
#SBATCH --error=/usr/stud/faragy/storage/user/logs/SSG_v3.12/%j_error.out

# Activate everything you need
#module load cuda/11.3
#Start Training From checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset "/usr/stud/faragy/storage/user/data/lmdb_datasets/lmdb_v3.6" --inception "/usr/stud/faragy/storage/user/data/inception_models/inception_v3.6.pkl" --save_every 5000  --checkpoint_dir /usr/stud/faragy/storage/user/data/checkpoints/SSG_v3.12 --seg_dim 16 --size 256  --residual_refine 
#Preprocessing
#python ~/storage/user/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/usr/stud/faragy/storage/user/data/cityscapes/gtFine" --output=/usr/stud/faragy/storage/user/data/preprocessed/v3.6/gtFine_preprocessed/
#Prepare lmdb  Data for 5k images
#python prepare_mask_data.py --cityscapes "True" /usr/stud/faragy/storage/user/data/cityscapes/leftImg8bit /usr/stud/faragy/storage/user/data/preprocessed/v3.6/gtFine_preprocessed --out /usr/stud/faragy/storage/user/data/lmdb_datasets/lmdb_v3.6 --size 256
#Training Inception Network
#python prepare_inception.py "/usr/stud/faragy/storage/user/data/lmdb_datasets/lmdb_v3.6" --output "/usr/stud/faragy/storage/user/data/inception_models/inception_v3.6.pkl" --size 256 --dataset_type mask


##!!!!!!!!!!!!!!!!!!OLD !!!!!!!!!!!!!!!
## Check FSD for 1 checkpoint:
#python calc_fsd.py --ckpt "/no_backups/g013/checkpoints/version_3/ckpt/045000.pt" --dataset="/no_backups/g013/data/v_2" --real_dataset_values="/no_backups/g013/data/real_dataset_cond.npy" --sample=1000
#Preprocess CityScapes
#python ~/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/data/public/cityscapes/gtFine" --output="/no_backups/g013/data/v3_no_test/"
#Load From Checkpoints
#python train.py --dataset /no_backups/g013/data/lmdb_cityscapes_256 --inception /no_backups/g013/data/inception_cityscapes_256.pkl --save_every 5000 --ckpt /no_backups/g013/checkpoints/cityscapes_256/ckpt/200000.pt --checkpoint_dir /no_backups/g013/checkpoints/cityscapes_256 --seg_dim 9 --size 256 --transparent_dims 3 --residual_refine 

#Generate
#python visualize/generate.py /no_backups/g013/checkpoints/version_3/ckpt/045000.pt --outdir results/samples --sample 20