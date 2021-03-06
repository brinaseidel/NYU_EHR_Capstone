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
from transformers import (XLNetForSequenceClassification, XLNetConfig)
from xlnet_evaluation import (get_batched_preds, get_combined_eval_metrics,  macroAUC, topKPrecision)
import random
import torch.optim as optim
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from sklearn import metrics
import json
import pickle
import math

def load_summarized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/', batch=False):
	
	# Read in the correct batch
	if batch:
		input_summaries = torch.load(os.path.join(feature_save_path, set_type + '_summaries_{}.pt'.format(batch)))
	else:
		input_summaries = torch.load(os.path.join(feature_save_path, set_type + '_summaries.pt'))
	# Read in the corresponding labels
	if batch:
		label_ids = torch.load(os.path.join(feature_save_path, set_type + '_doc_label_ids_{}.pt'.format(batch)))
	else:
		label_ids = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
	data = TensorDataset(input_summaries, label_ids)

	# Note: Possible to use SequentialSampler for eval, run time might be better
	sampler = RandomSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = False)

	return dataloader

def set_seeds(seed, n_gpu):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)

class SlidingClassifier(torch.nn.Module):
	def __init__(self, num_layers, hidden_size, p, input_size=768, num_codes=2292, activation_function='sigmoid'):
		super(SlidingClassifier, self).__init__()
		self.activation_function = activation_function

		self.transform_to_hidden = nn.Linear(input_size,hidden_size)
		self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
						for _ in range(num_layers)])
		if activation_function == 'sigmoid':
			self.activation = nn.Sigmoid()
		self.dropouts = nn.ModuleList([nn.Dropout(p=p)
						for _ in range(num_layers)])
		self.projection = nn.Linear(hidden_size, num_codes)

	def forward(self, inp):
		output = self.transform_to_hidden(inp)
		for dropout, linear in zip(self.dropouts, self.transforms):
			output = linear(output)
			if self.activation_function is not None:
				output = self.activation(output)
			output = dropout(output)
		output = self.projection(output)
		return output

def train(model, optimizer, num_train_epochs, n_gpu, device, model_id, models_folder = '/gpfs/data/razavianlab/capstone19/models', save_step = 100000, train_logging_step = 1000, val_logging_step = 100000, eval_folder = '/gpfs/data/razavianlab/capstone19/evals', n_saved_train_batches=1, n_saved_val_batches=1, batch_size=32, feature_save_path='/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
	global_step = 0

	# Get path to the file where we will save train performance
	train_file_name = os.path.join(eval_folder, model_id + "_train_metrics.p")
	# Create empty data frame to store training results in (to be written to train_file_name)
	train_results = pd.DataFrame(columns=['loss'])

	# Get path to the file where we will save validation performance
	val_file_name = os.path.join(eval_folder, model_id + "_val_metrics.p")
	# Create empty data frame to store evaluation results in (to be written to val_file_name)
	val_results = pd.DataFrame(columns=['loss', 'micro_AUC', 'macro_AUC', 'top1_precision', 'top3_precision', 'top5_precision', 'micro_f1' ,'macro_f1', 'macro_AUC_list' ])

	saved_batch_order = list(range(1, 1+n_saved_train_batches))

	# Create folder to save all checkpoints for this model
	model_save_path = os.path.join(models_folder, model_id)
	if not os.path.exists(model_save_path):
		os.makedirs(model_save_path)

	for epoch in range(num_train_epochs):
		train_loss = 0
		number_steps = 0
		# Shuffle the order of the saved data
		random.shuffle(saved_batch_order)
		for saved_batch in saved_batch_order:
			train_dataloader = [] # to avoid storing multiple dataloaders in memory at the same time
			train_dataloader = load_summarized_examples(batch_size, set_type = "train", feature_save_path=feature_save_path, batch=saved_batch)
				
			for i, batch in enumerate(train_dataloader):

				model.train()
				input_summaries, label_ids = batch

				input_summaries = input_summaries.to(device).float()

				# Might need to add .half() or .long() depending on amp versions
				label_ids = label_ids.to(device).float()

				logits = model(inp=input_summaries)

				criterion = BCEWithLogitsLoss()
				loss = criterion(logits, label_ids)

				if n_gpu > 1:
					loss = loss.mean()

				train_loss += loss.item()
				number_steps += 1
				mean_loss = train_loss/number_steps

				loss.backward()
				optimizer.step()
				model.zero_grad()

				global_step += 1

				# Log training loss
				if train_logging_step > 0 and global_step % train_logging_step == 0:
					logger.info("Training loss (Epoch {0}, Global Step {1}): {2:.5f}".format(epoch, global_step, mean_loss))
					train_results = train_results.append(pd.DataFrame({'loss': mean_loss}, index=[global_step]))
					pickle.dump(train_results, open(train_file_name, "wb"))
					os.system("chgrp razavianlab {}".format(train_file_name))
					#os.chmod(train_file_name, stat.S_IRWXG)
				# Log validtion metrics
				if val_logging_step > 0 and global_step % val_logging_step == 0:
					logger.info("Running validation")
					eval_losses = []
					number_steps_list = []
					targets = np.empty([0, 2292])
					predictions = np.empty([0, 2292])
					for saved_val_batch in range(1, 1+n_saved_val_batches):
						val_dataloader = [] # to avoid storing multiple dataloaders in memory at the same time
						val_dataloader = load_summarized_examples(batch_size, set_type = "val", feature_save_path=feature_save_path, batch=saved_val_batch)
						eval_loss, number_steps, target, pred = get_batched_preds(dataloader = val_dataloader, model = model, model_id = model_id, n_gpu=n_gpu, device=device, sliding_window=True)
						eval_losses.append(eval_loss)
						number_steps_list.append(number_steps)
						targets = np.concatenate((targets, target))
						predictions = np.concatenate((predictions, pred))
					results = get_combined_eval_metrics(dataloader = val_dataloader, model = model, model_id = model_id,  eval_losses=eval_losses, number_steps=number_steps_list, preds=predictions, target=targets, n_gpu=n_gpu, device=device, sliding_window=True)
					val_results = val_results.append(pd.DataFrame(results, index=[global_step]))
					pickle.dump(val_results, open(val_file_name, "wb"))
					os.system("chgrp razavianlab {}".format(val_file_name))
				# Save a copy of the model every save_step
				if save_step > 0 and global_step % save_step == 0:
					# Save model and optimizer checkpoints
					checkpoint_save_path = os.path.join(model_save_path, 'model_checkpoint_{}'.format(int(global_step/save_step)))

					if not os.path.exists(checkpoint_save_path):
						os.makedirs(checkpoint_save_path)

					torch.save({'model':model.state_dict(),
								'optimizer':optimizer.state_dict(),
								'step': global_step}, os.path.join(checkpoint_save_path, 'model.pt'))
					logger.info("Saving model checkpoint to {}".format(checkpoint_save_path))

	# Save model and optimizer checkpoints
	final_save_path = os.path.join(model_save_path, 'model_checkpoint_final')

	if not os.path.exists(final_save_path):
		os.makedirs(final_save_path)

	torch.save({'model':model.state_dict(),
				'optimizer':optimizer.state_dict(),
				'step': global_step}, os.path.join(final_save_path, 'model.pt'))
	logger.info("Saving final model to {}".format(final_save_path))


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

	parser.add_argument("--batch_size",
						default=32,
						type=int,
						help="Indicate batch size")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--num_hidden_layers',
						type=int,
						default=5,
						help="Number of hidden layers for MLP classifier")
	parser.add_argument('--hidden_size',
						type=int,
						default=1024,
						help="Hidden size for MLP classifier")
	parser.add_argument("--drop_rate",
						default=0.3,
						type=float,
						help="Droprate in between hidden layers for MLP classifer")
	parser.add_argument("--activation_function",
						default='sigmoid',
						type=str,
						help="Activation function for MLP classifer")
	parser.add_argument("--learning_rate",
						default=0.0001,
						type=float,
						help="Learning rate for training models")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--val_logging_step",
						default=100000,
						type=int,
						help="Number of steps in between logs of performance on validation set")
	parser.add_argument("--train_logging_step",
						default=1000,
						type=int,
						help="Number of steps in between logs of performance on training set")
	parser.add_argument("--save_step",
						default=100000,
						type=int,
						help="Number of steps to save model parameters")
	parser.add_argument("--model_id",
						type=str,
						help="Model and optimizer will be saved at '/gpfs/data/razavianlab/capstone19/models/model_id'. ")
	parser.add_argument("--feature_save_dir",
						type=str,
						help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/feature_save_dir'. ")
	parser.add_argument("--batched_data",
						type=bool,
						help="True/False indicating whether the summarized data has been saved in batches")
	parser.add_argument("--n_saved_train_files",
						type=int,
						help="Saved training data has been split into this many files'. ")
	parser.add_argument("--n_saved_val_files",
						type=int,
						help="Saved validation data has been split into this many files'. ")
	args = parser.parse_args()

	if args.activation_function == 'None':
		args.activation_function = None

	# Set random seed
	set_seeds(seed = args.seed, n_gpu = n_gpu)

	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)

	# Initialize classifier
	model = SlidingClassifier(num_layers=args.num_hidden_layers, hidden_size=args.hidden_size, p=args.drop_rate, activation_function=args.activation_function)
	model.to(device)
	model_parameters = [p for p in model.parameters()]
	optimizer = optim.Adam(model_parameters, lr=args.learning_rate)

	logger.info("***** Running training *****")
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Total train batch size  = %d", args.batch_size)

	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
	train(model = model, optimizer = optimizer, num_train_epochs = args.num_train_epochs, n_gpu = n_gpu, device = device,  model_id = args.model_id, save_step = args.save_step, train_logging_step = args.train_logging_step, val_logging_step = args.val_logging_step, n_saved_train_batches=args.n_saved_train_files, n_saved_val_batches=args.n_saved_val_files,batch_size=args.batch_size,  feature_save_path=feature_save_path)


if __name__ == "__main__":
	main()
