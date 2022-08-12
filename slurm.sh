#!/bin/bash -l
#SBATCH --job-name="SemanticStyleGan_CityScapes"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=4-23:00:00
#SBATCH --mail-type=ALL
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=./log_files/SSG%j.%N_output.out
#SBATCH --error=./log_files/SSG%j.%N_error.out
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.3
#Start Training From the Beginning
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_cityscapes_256_version_3_no_test --inception /no_backups/g013/data/inception_cityscapes_256_version_3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/version_3_no_test --seg_dim 16 --size 256  --residual_refine 

## Check FSD for 1 checkpoint:
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/version_3/ckpt/045000.pt" --dataset="/no_backups/g013/data/v_2" --real_dataset_values="/no_backups/g013/data/real_dataset_cond.npy" --sample=1000
#Preprocess CityScapes
python3.9 ~/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/data/public/cityscapes/gtFine" --output="/no_backups/g013/data/v3_no_test/"
#Load From Checkpoints
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_cityscapes_256 --inception /no_backups/g013/data/inception_cityscapes_256.pkl --save_every 5000 --ckpt /no_backups/g013/checkpoints/cityscapes_256/ckpt/200000.pt --checkpoint_dir /no_backups/g013/checkpoints/cityscapes_256 --seg_dim 9 --size 256 --transparent_dims 3 --residual_refine 
#Training Inception Network
#python3.9 prepare_inception.py /no_backups/g013/data/lmdb_cityscapes_256_version_3_no_test --output /no_backups/g013/data/inception_cityscapes_256_version_3.pkl --size 256 --dataset_type mask
#Prepare Data for 5k images
#python3.9 prepare_mask_data.py --cityscapes "True" /no_backups/g013/data/leftImg8bit /no_backups/g013/data/v3_no_test --out /no_backups/g013/data/lmdb_cityscapes_256_version_3_no_test --size 256
#Generate
#python3.9 visualize/generate.py /no_backups/g013/checkpoints/version_3/ckpt/045000.pt --outdir results/samples --sample 20