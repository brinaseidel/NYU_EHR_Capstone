#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=5:00:00
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:2
#SBATCH --output=/gpfs/data/razavianlab/capstone19/logs/preprocess_full_512_val_test.log

module load python/gpu/3.6.5
module load gcc/4.9.3

pip install --user transformers

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
#python xlnet_preprocess.py --filename "train_cleaned.csv.gz" --set_type "train" --max_seq_length 512 --feature_save_dir "full_512"
python xlnet_preprocess.py --filename "dev_cleaned.csv.gz" --set_type "val" --max_seq_length 512 --feature_save_dir "full_512"
python xlnet_preprocess.py --filename "test_cleaned.csv.gz" --set_type "test" --max_seq_length 512 --feature_save_dir "full_512"
