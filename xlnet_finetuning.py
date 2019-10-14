import torch
import logging
import argparse
import os
import sys
import pandas as pd
import numpy as np
import tensorboardX
from apex import amp
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam
logger = logging.getLogger(__name__)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import (XLNetForSequenceClassification, XLNetConfig)

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
	set_seeds(seed = args.seed, n_gpu = n_gpu)

	# Parse arguments
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
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
	args = parser.parse_args()

	# Load data
	logger.info("Loading train dataset")
	train_dataloader = load_featurized_examples(args.batch_size, set_type = "train")
	logger.info("Loading validation dataset")
	val_dataloader = load_featurized_examples(args.batch_size, set_type = "val")
	
	# Load pretrained model
	num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
	config = XLNetConfig.from_pretrained('xlnet-base-cased', num_labels=2) # TODO: check if we need this
	model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
	model.to(device)
	
	tb_writer = SummaryWriter()

	# TODO: understand this code
	no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=4e-5, eps=1e-8)
    if args.fp16:
    	warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    	model, optimizer = amp.initialize(model, optimizer, opt_level= 'O1')
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=len(train_dataloader)*args.num_train_epochs)

    # STOPPED AT logger.info("***** Running training *****")

 


