#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:2
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/pretrained_embedding_full_256.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
python xlnet_pretrained_embedding.py --feature_save_dir "full_train_256" --set_type "train"
python xlnet_pretrained_embedding.py --feature_save_dir "full_train_256" --set_type "val"
