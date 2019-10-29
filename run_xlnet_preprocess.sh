#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:2
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/preprocess_small_256.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
python xlnet_preprocess.py --filename "small.csv.gz" --set_type "train" --max_seq_length 256 --feature_save_dir "small_256"
python xlnet_preprocess.py --filename "small.csv.gz" --set_type "val" --max_seq_length 256 --feature_save_dir "small_256"
