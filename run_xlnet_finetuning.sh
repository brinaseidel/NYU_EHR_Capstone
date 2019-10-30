#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=9-5:00:00
#SBATCH --partition=gpu8_long
#SBATCH --gres=gpu:2
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/xlnet_finetune_full_256.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"

#python xlnet_finetuning.py --model_id xlnet_finetune --batch_size 32 --num_train_epochs 3 --seed 100 --fp16
#python -u xlnet_finetuning.py --model_id xlnet_finetune_small256 --batch_size 32 --num_train_epochs 10 --seed 100 --fp16 --train_logging_step 1 --val_logging_step 10 --save_step 100000 --feature_save_dir small_256
python xlnet_finetuning.py --model_id xlnet_finetune_full_256 --batch_size 32 --num_train_epochs 3 --seed 100 --fp16 --train_logging_step 1000 --val_logging_step 10000 --save_step 100000 --feature_save_dir full_train_256



