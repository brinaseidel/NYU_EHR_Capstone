import torch
import logging
import argparse
import os
import sys
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/ehr_transformer_xlnet/'):
	input_ids = torch.load(feature_save_path + set_type + '_input_ids.pt')
	input_mask = torch.load(feature_save_path + set_type + '_input_mask.pt') 
	segment_ids = torch.load(feature_save_path + set_type + '_segment_ids.pt')
	labels = torch.load(feature_save_path + set_type + '_labels.pt')
	data = TensorDataset(input_ids, input_mask, segment_ids, labels)

	# Note: Possible to use SequentialSampler for eval, run time might be better
	if torch.cuda.is_available():
		sampler = DistributedSampler(data)
	else:
		sampler = RandomSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = True)

	return dataloader

def set_seeds(seed, n_gpu):
	random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main():

	# Section: Set device for PyTorch
	if torch.cuda.is_available():
		 # might need to update when using more than 1 GPU
		device = torch.device("cuda") 
		torch.distributed.init_process_group(backend='nccl')
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device("cpu")
		n_gpu = 0

	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Indicate batch size")

	parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

	parser.add_argument("--num_train_epochs",
	                    default=3.0,
	                    type=float,
	                    help="Total number of training epochs to perform.")

	args = parser.parse_args()

	logger.info("Loading train dataset")
	train_dataloader = load_featurized_examples(args.batch_size, set_type = "train")
	logger.info("Loading validation dataset")
	val_dataloader = load_featurized_examples(args.batch_size, set_type = "val")
	set_seeds(seed = args.seed, n_gpu = n_gpu)

	num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
	





