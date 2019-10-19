import torch
import logging
import argparse
import os
import sys
import pandas as pd
import numpy as np
from apex import amp
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam
logger = logging.getLogger(__name__)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import (XLNetForSequenceClassification, XLNetConfig)
from xlnet_evaluation import evaluate
import random

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/'):
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

def initialize_optimizer(model, args):
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
		]
	optimizer = AdamW(optimizer_grouped_parameters, lr=4e-5, eps=1e-8)
	model, optimizer = amp.initialize(model, optimizer, opt_level= 'O1')
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=len(train_dataloader)*args.num_train_epochs)
	return optimizer, scheduler, model

def train(train_dataloader, val_dataloader, model, optimizer, scheduler, num_train_epochs, n_gpu, model_id, models_folder = '/gpfs/data/razavianlab/capstone19/models', save_step = 100000, logging_step = 100000):
	global_step = 0
	# create folder to save all checkpoints for this model
	model_save_path = os.path.join(models_folder, model_id)
	if not os.path.exists(model_save_path):
		os.makedirs(model_save_path)

	for epoch in range(num_train_epochs):
		with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as progressbar:
			train_loss = 0
			number_steps = 0
			for i, batch in enumerate(train_dataloader):
				model.train()
				input_ids, input_mask, segment_ids, label_ids = batch

				input_ids = input_ids.to(device).long()
				input_mask = input_mask.to(device).long()
				segment_ids = segment_ids.to(device).long()

				# Might need to add .half() or .long() depending on amp versions
				label_ids = label_ids.to(device)

				logits = model(input_ids, segment_ids, input_mask, labels=None)

				criterion = BCEWithLogitsLoss()
				loss = criterion(logits, label_ids)

				if n_gpu > 1:
					loss = loss.mean()

				# Required for fp16
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
				# Prevent exploding gradients and set max gradient norm to 1
				torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), max_norm = 1)

				train_loss += loss.item()
				number_steps += 1
				mean_loss = train_loss/number_steps

				# TODO: Calculate training loss after some specified number of batches
				progressbar.update(1)
				progressbar.set_postfix_str("Loss: {.5f}".format(mean_loss))

				scheduler.step()
				optimizer.step()
				model.zero_grad()

				global_step += 1

				if logging_step > 0 and global_step % logging_step == 0:
					# Log metrics
					eval_file_name = model_id + '_checkpoint_{}.json'.format(global_step % logging_step)
					results = evaluate(dataloader = val_dataloader, model = model, eval_file_name = eval_file_name)

				if save_step > 0 and global_step % save_step == 0:
					# Save model and optimizer checkpoints
					checkpoint_save_path = os.path.join(model_save_path, 'model_checkpoint_{}'.format(global_step % save_step))
					
					if not os.path.exists(checkpoint_save_path):
						os.makedirs(checkpoint_save_path)

					model_to_save = model.module if hasattr(model, 'module') else model
					model_to_save.save_pretrained(checkpoint_save_path)
					torch.save({'optimizer':optimizer.state_dict(),
								'scheduler': scheduler.state_dict(),
								'step': global_step}, os.path.join(checkpoint_save_path, 'optimizer.pt'))
					logger.info("Saving model checkpoint to {}".format(checkpoint_save_path))

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
	parser.add_argument("--logging_step",
						default=100000,
						type=int,
						help="Number of steps to log progress")
	parser.add_argument("--save_step",
						default=100000,
						type=int,
						help="Number of steps to save model parameters")
	parser.add_argument("--model_id",
						type=str,
						help="Model and optimizer will be saved at '/gpfs/data/razavianlab/capstone19/models/model_id'. ")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	args = parser.parse_args()

	# Set random seed
	set_seeds(seed = args.seed, n_gpu = n_gpu)


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

	optimizer, scheduler, model = initialize_optimizer(model, args)

	logger.info("***** Running training *****")
	logger.info("  Num batches = %d", len(train_dataloader))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Total train batch size  = %d", args.batch_size)
	logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)

	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
	train(train_dataloader = train_dataloader, val_dataloader = val_dataloader, model = model, optimizer = optimizer, scheduler = scheduler, num_train_epochs = args.num_train_epochs, n_gpu = n_gpu, model_id = args.model_id, save_step = args.save_step, logging_step = args.logging_step)

if __name__ == "__main__":
	main()
