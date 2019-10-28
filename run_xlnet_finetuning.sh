#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:00:00
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/xlnet_finetune_256_small.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"

#python xlnet_finetuning.py --model_id xlnet_finetune --batch_size 32 --num_train_epochs 3 --seed 100 --fp16
python -u xlnet_finetuning.py --model_id xlnet_finetune_256_small --batch_size 32 --num_train_epochs 3 --seed 100 --fp16 --train_logging_step 50 --val_logging_step 10000 --save_step 100000 --feature_save_dir small_256

