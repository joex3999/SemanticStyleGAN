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
#SBATCH --output=./log_files/control/SSG%j.%N_output.out
#SBATCH --error=./log_files/control/SSG%j.%N_error.out
#SBATCH --qos=batch
#SBATCH --exclude=linse9


# Activate everything you need
module load cuda/11.3

#Salloc command
#salloc --job-name="SemanticStyleGan_CityScapes" --nodes=1 --ntasks=1  --cpus-per-task=2 --mem=32G --gpus=1 --time=4-23:00:00 --qos=batch


python3.9 visualize/generate.py /no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt --outdir results/controlled_samples --sample 100 --save_latent

#python3.9 visualize/generate_components.py /no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt --outdir results/samples --latent results/controlled_samples/000002_latent.npy

#python3.9 visualize/generate_video.py /no_backups/g013/checkpoints/SSG_v3.13/ckpt/140000.pt --outdir results/video --latent results/saved_samples/second_latent.npy
