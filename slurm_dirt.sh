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
#SBATCH --output=./log_files/SBGAN/SBGAN%j.%N_output_33C_2.out
#SBATCH --error=./log_files/SBGAN/SBGAN%j.%N_error_33C_2.out
#SBATCH --qos=batch
#SBATCH --exclude=linse10
# Activate everything you need
module load cuda/11.3



##FID Calculation
#python3.9 calc_fid.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --inception /no_backups/g013/data/inception_models/inception_v3.3_for_fid_3k.pkl --n_sample=3000
python3.9 calc_fid.py  --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --image_directory "/no_backups/g013/other_GANs/SBGAN_cityscapes_imgs" --max_image_number 24999
## Testing with SSG
#python3.9 calc_fid.py  --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --image_directory "/no_backups/g013/other_GANs/ssg_cityscapes_imgs" --max_image_number 1000



##MIOU ##Execute with myenv_2 env.
##For test
# python3.9  calc_test_miou_dirt.py \
#  --name test_synthetic_cityscapes_128 \
#   --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --sample 5000 --res_semantics 17\
#  --dataset "cityscapes" --max_dim 1024 --dim 128 \
#  --x_model deeplabv3 --x_which_iter 137 --x_load_path "/no_backups/g013/checkpoints/other_checkpoints/segmenter_real_cityscapes25k_128" \
#  --batch_size 16 \
#  --x_synthetic_dataset\
#  --lmdb "/no_backups/g013/data/lmdb_datasets/lmdb_v3.3"\
#  --train_miou

#python3.9 calc_fid_dirt.py --dataset="/no_backups/g013/data/leftImg8bit" --sample=20_000

# python3.9  calc_test_miou.py \
#  --name test_synthetic_cityscapes_128 \
#   --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --sample 5000 --res_semantics 17\
#  --dataset "cityscapes" --max_dim 1024 --dim 128 \
#  --x_model deeplabv3 --x_which_iter 137 --x_load_path "/no_backups/g013/checkpoints/other_checkpoints/segmenter_real_cityscapes25k_128" \
#  --batch_size 16 \
#  --x_synthetic_dataset\
#  --lmdb "/no_backups/g013/data/lmdb_datasets/lmdb_v3.3"\
#  --train_miou
