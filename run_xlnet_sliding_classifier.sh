#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu8_medium
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/xlnet_pretrained_small_256.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"

#python xlnet_finetuning.py --model_id xlnet_finetune --batch_size 32 --num_train_epochs 3 --seed 100 --fp16
#python -u xlnet_finetuning.py --model_id xlnet_finetune_small256 --batch_size 32 --num_train_epochs 10 --seed 100 --fp16 --train_logging_step 1 --val_logging_step 10 --save_step 100000 --feature_save_dir small_256
python xlnet_sliding_classifier.py --model_id xlnet_pretrained_small_256 --batch_size 32 --seed 100 \
    --num_hidden_layers 0 --hidden_size 1024 --drop_rate 0 --learning_rate 0.0001 \
    --num_train_epochs 10 --train_logging_step 10 \
    --val_logging_step 10 --save_step 50 --feature_save_dir small_256
