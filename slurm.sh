#!/bin/bash -l
#SBATCH --job-name="SemanticStyleGan_CityScapes"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mem=8G
#SBATCH --gpus=2
#SBATCH --time=4-23:00:00
#SBATCH --mail-type=ALL
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=./log_files/classes_ablation/SSG%j.%N_output_27C.out
#SBATCH --error=./log_files/classes_ablation/SSG%j.%N_error_27C.out
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.3

#Preprocess CityScapes
#p#ython3.9 ~/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/no_backups/g013/data/gtFine" --output="/no_backups/g013/data/preprocessed/v9_27C/"


#Prepare Data
#python3.9 prepare_mask_data.py --cityscapes "True" /no_backups/g013/data/leftImg8bit /no_backups/g013/data/preprocessed/v9_27C --out /no_backups/g013/data/lmdb_datasets/lmdb_v9_27C --size 256
#python3.9 prepare_inception.py /no_backups/g013/data/lmdb_datasets/lmdb_v9_27C --output /no_backups/g013/data/inception_models/inception_v9_27C.pkl --size 256 --dataset_type mask

CUDA_VISIBLE_DEVICES=0,1 python3.9 -m torch.distributed.launch --nproc_per_node=2 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v9_27C --inception /no_backups/g013/data/inception_models/inception_v9_27C.pkl --save_every 10000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_27C_new  --seg_dim 27 --size 256  --residual_refine --batch=1 --ckpt "/no_backups/g013/checkpoints/SSG_27C/ckpt/010000.pt" 


#Training using Multiple GPUs
#CUDA_VISIBLE_DEVICES=0,1 python3.9 -m torch.distributed.launch --nproc_per_node=2 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v6_24C --inception /no_backups/g013/data/inception_models/inception_v6_24C.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v6.0  --seg_dim 24 --size 256  --residual_refine --batch=2
#CUDA_VISIBLE_DEVICES=0,1,2 python3.9 -m torch.distributed.launch --nproc_per_node=3 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v6_24C --inception /no_backups/g013/data/inception_models/inception_v6_24C.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v7.0 --ckpt "/no_backups/g013/checkpoints/SSG_v7.0/ckpt/045000.pt" --seg_dim 24 --size 256  --residual_refine --batch=4

#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.10  --seg_dim 16 --size 256  --residual_refine

#Start Training From the Beginning
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.5_2  --seg_dim 16 --size 256  --residual_refine
#CUDA_VISIBLE_DEVICES=0,1,2 python3.9 -m torch.distributed.launch --nproc_per_node=3 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --checkpoint_dir /no_backups/g013/checkpoints/SSG_v4.1  --seg_dim 16 --size 256  --residual_refine
##Inverting an image
#python3.9 visualize/invert.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --imgdir "inverting_input" --outdir "inverting_output"
#Preprocess CityScapes
#python3.9 ~/SemanticStyleGAN/data/preprocess_cityscapes.py --data="/no_backups/g013/data/gtFine" --output="/no_backups/g013/data/preprocessed/v6_24C/"

#Load From Checkpoints
#python3.9 train.py --dataset /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --inception /no_backups/g013/data/inception_models/inception_v3.3.pkl --save_every 5000 --ckpt /no_backups/g013/checkpoints/SSG_v3.3/ckpt/200000.pt --checkpoint_dir /no_backups/g013/checkpoints/SSG_v3.3 --seg_dim 9 --size 256 --transparent_dims 3 --residual_refine 

#Prepare Data
#python3.9 prepare_mask_data.py --cityscapes "True" /no_backups/g013/data/leftImg8bit /no_backups/g013/data/preprocessed/v6_24C --out /no_backups/g013/data/lmdb_datasets/lmdb_v6_24C --size 256

#Training Inception Network
#python3.9 prepare_inception.py /no_backups/g013/data/lmdb_datasets/lmdb_v6_24C --output /no_backups/g013/data/inception_models/inception_v6_24C.pkl --size 256 --dataset_type mask

##Calculate FSD:
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --dataset="/no_backups/g013/data/preprocessed/v3.3/gtFine_preprocessed" --real_dataset_values="./real_dataset_cond_old.npy" --save_real_dataset "True" --sample=50000
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/SSG_v4.1/ckpt/215000.pt" --dataset="/no_backups/g013/data/preprocessed/v3.3/gtFine_preprocessed" --real_dataset_values="./real_dataset_cond_old.npy" --save_real_dataset "True" --sample=50000

#Calculate KID:
#python3.9 calc_kid.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --dataset="/no_backups/g013/data/leftImg8bit" --sample=1000

##FOR CALCULATING FID5k ###
#Training Inception Network
#python3.9 prepare_inception.py /no_backups/g013/data/lmdb_datasets/lmdb_v3.3 --output /no_backups/g013/data/inception_models/inception_v3.3_for_fid_3k.pkl --size 256 --dataset_type mask --n_sample=3000

##Calculate FID:
#python3.9 calc_fid.py --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --inception /no_backups/g013/data/inception_models/inception_v3.3_for_fid_3k.pkl --n_sample=3000

##MIOU ##Execute with myenv_2 env.
##For test
# python3.9  calc_test_miou.py \
#  --name test_synthetic_cityscapes_128 \
#   --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --sample 5000 --res_semantics 17\
#  --dataset "cityscapes" --max_dim 1024 --dim 128 \
#  --x_model deeplabv3 --x_which_iter 137 --x_load_path "/no_backups/g013/checkpoints/other_checkpoints/segmenter_real_cityscapes25k_128" \
#  --batch_size 16 \
#  --x_synthetic_dataset\
#  --lmdb "/no_backups/g013/data/lmdb_datasets/lmdb_v3.3"\
#  --train_miou


## For Train 

# python3.9  calc_train_miou.py \
#  --name test_synthetic_cityscapes_128 \
#   --ckpt "/no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt" --sample 5000 --res_semantics 17\
#  --dataset "cityscapes" --max_dim 1024 --dim 128 \
#  --x_model deeplabv3 --x_which_iter 121 --x_load_path "/no_backups/g013/checkpoints/other_checkpoints/segmenter_synthetic_cityscapes_128_for_ssg" \
#  --batch_size 16 \
#  --x_synthetic_dataset\
#  --lmdb "/no_backups/g013/data/lmdb_datasets/lmdb_v3.3"\
#  --train_miou


##OLD !!!!!!1

## Check FSD for 1 checkpoint:
#python3.9 calc_fsd.py --ckpt "/no_backups/g013/checkpoints/version_3/ckpt/045000.pt" --dataset="/no_backups/g013/data/v_2" --real_dataset_values="/no_backups/g013/data/real_dataset_cond.npy" --sample=1000



#Generate
#python3.9 visualize/generate.py /no_backups/g013/checkpoints/version_3/ckpt/045000.pt --outdir results/samples --sample 20

#For using 2 gpus , up number of nodes:a40:1 and : 
#CUDA_VISIBLE_DEVICES=0,1  command 