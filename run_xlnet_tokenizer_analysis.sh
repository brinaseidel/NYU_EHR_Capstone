#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G
#SBATCH --time=10:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=/gpfs/data/razavianlab/capstone19/tokenizer_analysis/train_256_tokenizer_analysis.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
python xlnet_tokenizer_analysis.py --set_type "train" --feature_save_dir "full_train_256"
