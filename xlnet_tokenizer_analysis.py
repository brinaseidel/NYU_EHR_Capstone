import torch
import logging
import argparse
import os, stat
import sys
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import XLNetTokenizer
import random

def load_featurized_examples(set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
	input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
	input_mask = torch.load(os.path.join(feature_save_path, set_type + '_input_mask.pt'))
	segment_ids = torch.load(os.path.join(feature_save_path, set_type + '_segment_ids.pt'))
	labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
	data = TensorDataset(input_ids, input_mask, segment_ids, labels)

	# Note: Possible to use SequentialSampler for eval, run time might be better
	sampler = RandomSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=1, drop_last = True)

	return dataloader

def main():

	# Section: Set device for PyTorch
	if torch.cuda.is_available():
		# might need to update when using more than 1 GPU
		rank = 0
		torch.cuda.set_device(rank)
		device = torch.device("cuda", rank)
		#torch.distributed.init_process_group(backend='nccl')
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device("cpu")
		n_gpu = 0
        
	print("N GPU: ", n_gpu)
	# Parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument("--set_type",
						type=str,
						help="Specify train/val/test")
	parser.add_argument("--feature_save_dir",
						type=str,
						help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/feature_save_dir'. ")
	args = parser.parse_args()

	# Load data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading {} dataset".format(args.set_type))
	dataloader = load_featurized_examples(set_type = args.set_type, feature_save_path=feature_save_path)

	# Create XLNET pretrained tokenizer
	tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

	for i, batch in enumerate(dataloader):
		input_ids, input_mask, segment_ids, label_ids = batch
		input_ids = input_ids.tolist()[0]
		print(input_ids)
		print(tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))	
		individually_decoded = []
		for input_id in input_ids:
			individually_decoded.append(tokenizer.decode([input_id], clean_up_tokenization_spaces=False))
		print(individually_decoded)

if __name__ == "__main__":
	main()
