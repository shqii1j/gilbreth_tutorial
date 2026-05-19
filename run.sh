#!/bin/bash
#SBATCH --job-name=stat695_train
#SBATCH --account=statdept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH -p a30

# Activate environment
conda activate stat695

# Run training
python train.py --model resnet18 --epochs 10 --lr 0.01 --batch_size 128 --pretrained
