#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2-1:00:00
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:3
#SBATCH --output=/gpfs/data/razavianlab/capstone19/evals/full_evaluation_checkpoint1.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
python xlnet_evaluation.py --model_id xlnet_finetune_full_256 --checkpoint 1 --fp16 --feature_save_dir "test_256"
