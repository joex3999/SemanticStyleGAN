#!/bin/bash -l
#SBATCH --job-name="SemanticStyleGan_CityScapes"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=SSG%j.%N_output.out
#SBATCH --error=SSG%j.%N_error.out
#SBATCH --qos=batch
#SBATCH --mail-user=joex3999@gmail.com


# Activate everything you need
module load cuda/11.3

python3.9 train.py --dataset /no_backups/g013/data/lmdb_cityscapes_256 --inception /no_backups/g013/data/inception_cityscapes_256.pkl --checkpoint_dir /no_backups/g013/checkpoints/cityscapes_256 --seg_dim 9 --size 256 --transparent_dims 3 --residual_refine 
#python3.6 prepare_inception.py /no_backups/g013/data/lmdb_cityscapes_256 --output  /no_backups/g013/data/inception_cityscapes_256.pkl --size 256 --dataset_type mask

