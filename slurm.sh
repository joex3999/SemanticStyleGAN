#!/bin/bash
#SBATCH --job-name="Inf"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus=a40:1
#SBATCH --time=15:00:00
#SBATCH --mail-type=NONE
##SBATCH --partition=DEADLINE
##SBATCH --comment=ECCVRebuttal
#SBATCH --output=./slurmDebug/slurm-%j.out
#SBATCH --error=./slurmDebug/slurm-%j.out


python main_train.py
#python main_test.py
