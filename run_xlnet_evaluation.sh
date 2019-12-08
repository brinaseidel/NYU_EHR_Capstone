#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/razavianlab/capstone19/evals/xlnet_pretrained_full_256_lr1e-6_val_results.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"

python xlnet_evaluation.py --model_id xlnet_pretrained_full_256_lr1e-6 --checkpoint 3 --fp16 --feature_save_dir "full_train_256" --set_type "val" --model_type "classifier" --num_hidden_layers 0 --hidden_size 1024 --drop_rate 0


