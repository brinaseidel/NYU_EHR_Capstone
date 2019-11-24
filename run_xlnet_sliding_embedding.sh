#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/razavianlab/capstone19/sliding_embedding/sliding_embedding_512_small_sampling.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
python xlnet_sliding_embedding_v2.py --model_id xlnet_finetune_full_512_old_eval --checkpoint 3 --fp16 --feature_save_dir "small_train_sliding_512" --set_type "train"
