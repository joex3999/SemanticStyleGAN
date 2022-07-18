#!/bin/bash -l
#SBATCH --job-name="SemanticStyleGan_CityScapes"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus=a40:1
#SBATCH --time=15:00:00
#SBATCH --mail-type=NONE
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=SemanticStyleGan_CityScapes%j.%N_output.out
#SBATCH --error=SemanticStyleGan_CityScapes%j.%N_error.out
#SBATCH --qos=batch


# Activate everything you need
module load cuda/11.3

conda activate env
python main_train.py
#python main_test.py
