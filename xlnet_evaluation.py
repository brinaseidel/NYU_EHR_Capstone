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
from sklearn import metrics
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import json
import pickle
import torch.optim as optim
import torch.nn as nn

def load_featurized_examples(batch_size, set_type, sliding_window=False, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
	if sliding_window:
		input_summaries = torch.load(os.path.join(feature_save_path, set_type + '_summaries.pt'))
		labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
		data = TensorDataset(input_summaries, labels)
	else:
		input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
		input_mask = torch.load(os.path.join(feature_save_path, set_type + '_input_mask.pt'))
		segment_ids = torch.load(os.path.join(feature_save_path, set_type + '_segment_ids.pt'))
		labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
		data = TensorDataset(input_ids, input_mask, segment_ids, labels)

	# Note: Possible to use SequentialSampler for eval, run time might be better
	sampler = RandomSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = True)

	return dataloader

def macroAUC(pred, true):
	auc = []
	for i in range(pred.shape[1]):
		if (len(np.unique(true[:,i])) > 1) :
			auc.append(metrics.roc_auc_score(true[:,i], pred[:,i]))
		else:
			auc.append(0.5)
	return np.mean(auc), [auc]

def topKPrecision(pred, true, k):
	# pred: size of n_sample x n_class
	idx_sort = np.argsort(pred, axis=1)
	n_sample = true.shape[0]
	idx_row = np.array([int(x) for x in range(n_sample)]).reshape(-1, 1)
	true_sort = true[idx_row, idx_sort]
	result = np.sum(true_sort[:,-k:].astype(np.float64)) / k / n_sample
	return result

def evaluate(dataloader, model, model_id, n_gpu, device, sliding_window=False):
	logger.info("***** Running evaluation *****")
	logger.info("  Num batches = %d", len(dataloader))
	eval_loss = 0.0
	number_steps = 0
	preds = []
	target = []

	for batch in dataloader:
		model.eval()

		# If using the sliding window classifier, get processed data
		if sliding_window:
			input_summaries, label_ids = batch
			input_summaries = input_summaries.to(device).float()
			label_ids = label_ids.to(device).float()
		else:
			input_ids, input_mask, segment_ids, label_ids = batch
			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()
			label_ids = label_ids.to(device).float()

		with torch.no_grad():
			if sliding_window:
				logits = model(inp=input_summaries)
			else:
				logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
		criterion = BCEWithLogitsLoss()
		loss = criterion(logits, label_ids)

		# TODO: Check why we take mean
		#print("loss = ", loss)
		if n_gpu > 1:
			eval_loss += loss.mean().item()
		else:
			eval_loss += loss.item()

		number_steps += 1
		preds.append(torch.sigmoid(logits).detach().cpu()) # sigmoid returns probabilities
		target.append(label_ids.detach().cpu())
		# not used in calculations, for sanity checks
		mean_loss = eval_loss/number_steps

	eval_loss = eval_loss / number_steps
	preds = torch.cat(preds).numpy()
	target = torch.cat(target).byte().numpy()

	#np.set_printoptions(threshold=2292)
	#preds_when_true = np.ma.array(preds, mask=target)
	#preds_when_true = np.ma.filled(preds_when_true, np.nan)
	#preds_when_false = np.ma.array(preds, mask=target==0)
	#preds_when_false - np.ma.filled(preds_when_false, np.nan)
	#for quantile in range(1, 100):
#		print("Quantile {} of predicted values: {}".format(quantile, np.percentile(preds, quantile, axis=0)))
	#	print("Quantile {} of predicted values where target = True: {}".format(quantile, np.nanpercentile(preds_when_true, quantile, axis=0)))
	#	print("Quantile {} of predicted values where target = False: {}".format(quantile, np.nanpercentile(preds_when_false, quantile, axis=0)))
	micro_AUC = metrics.roc_auc_score(target, preds, average='micro')
	macro_AUC, macro_AUC_list = macroAUC(preds, target)
	top1_precision = topKPrecision(preds, target, k = 1)
	top3_precision = topKPrecision(preds, target, k = 3)
	top5_precision = topKPrecision(preds, target, k = 5)
	micro_f1 = {}
	macro_f1 = {}
	for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
		micro_f1['threshold {}'.format(threshold)] = metrics.f1_score(target, preds>threshold, average='micro')
		macro_f1['threshold {}'.format(threshold)] = metrics.f1_score(target, preds>threshold, average='macro')

	logger.info("Evaluation loss : {}".format(str(eval_loss)))
	logger.info("micro_AUC : {} ,  macro_AUC : {}".format(str(micro_AUC) ,str(macro_AUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))
	logger.info("micro_f1 : {} , macro_f1 : {}".format(str(micro_f1), str(macro_f1)))

	results = {
				'loss': eval_loss,
				'micro_AUC' : micro_AUC ,
				'macro_AUC' : macro_AUC,
				'top1_precision' : top1_precision ,
				'top3_precision' : top3_precision ,
				'top5_precision' : top5_precision,
				'micro_f1' : [micro_f1],
				'macro_f1' : [macro_f1],
				'macro_AUC_list' : macro_AUC_list
				}

	return results



def get_batched_preds(dataloader, model, model_id, n_gpu, device, sliding_window=False):
	eval_loss = 0.0
	number_steps = 0
	preds = []
	target = []

	for batch in dataloader:
		model.eval()

		# If using the sliding window classifier, get processed data
		if sliding_window:
			input_summaries, label_ids = batch
			input_summaries = input_summaries.to(device).float()
			label_ids = label_ids.to(device).float()
		else:
			input_ids, input_mask, segment_ids, label_ids = batch
			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()
			label_ids = label_ids.to(device).float()

		with torch.no_grad():
			if sliding_window:
				logits = model(inp=input_summaries)
			else:
				logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
		criterion = BCEWithLogitsLoss()
		loss = criterion(logits, label_ids)

		# TODO: Check why we take mean
		#print("loss = ", loss)
		if n_gpu > 1:
			eval_loss += loss.mean().item()
		else:
			eval_loss += loss.item()

		number_steps += 1
		preds.append(torch.sigmoid(logits).detach().cpu()) # sigmoid returns probabilities
		target.append(label_ids.detach().cpu())
	preds = torch.cat(preds).numpy()
	target = torch.cat(target).byte().numpy()

	return eval_loss, number_steps, target, preds

def get_combined_eval_metrics(dataloader, model, model_id, eval_losses, number_steps, preds, target, n_gpu, device, sliding_window=False):
	'''For data that has been saved in batches'''
	eval_loss = sum(eval_losses)/ sum(number_steps)
	micro_AUC = metrics.roc_auc_score(target, preds, average='micro')
	macro_AUC, macro_AUC_list = macroAUC(preds, target)
	top1_precision = topKPrecision(preds, target, k = 1)
	top3_precision = topKPrecision(preds, target, k = 3)
	top5_precision = topKPrecision(preds, target, k = 5)
	micro_f1 = {}
	macro_f1 = {}
	for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
		micro_f1['threshold {}'.format(threshold)] = metrics.f1_score(target, preds>threshold, average='micro')
		macro_f1['threshold {}'.format(threshold)] = metrics.f1_score(target, preds>threshold, average='macro')


	logger.info("Evaluation loss : {}".format(str(eval_loss)))
	logger.info("micro_AUC : {} ,  macro_AUC : {}".format(str(micro_AUC) ,str(macro_AUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))
	logger.info("micro_f1 : {} , macro_f1 : {}".format(str(micro_f1), str(macro_f1)))

	results = {
				'loss': eval_loss,
				'micro_AUC' : micro_AUC ,
				'macro_AUC' : macro_AUC,
				'top1_precision' : top1_precision ,
				'top3_precision' : top3_precision ,
				'top5_precision' : top5_precision,
				'micro_f1' : [micro_f1],
				'macro_f1' : [macro_f1],
				'macro_AUC_list' : macro_AUC_list
				}

	return results



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

def main():

	# Set device for PyTorch
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

	parser.add_argument("--model_id",
						type=str,
						help="Model and optimizer should be saved at a folder inside '/gpfs/data/razavianlab/capstone19/models/{model_id}'. ")
	parser.add_argument("--checkpoint",
					type=str,
					help="Checkpoint number. Model and optimizer should be saved at '/gpfs/data/razavianlab/capstone19/models/{model_id}/model_checkpoint_{checkpoint}'. ")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument("--feature_save_dir",
						type=str,
						help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/{feature_save_dir}'. ")
	parser.add_argument("--set_type",
						type=str,
						help="Specify train/val/test")
	parser.add_argument("--model_type",
						type=str,
						default='xlnet',
						help="Specify xlnet or classifier")
	parser.add_argument('--num_hidden_layers',
						type=int,
						default=5,
						help="Number of hidden layers for MLP classifier (not needed to evaluate a model with XLNet architecture)")
	parser.add_argument('--hidden_size',
						type=int,
						default=1024,
						help="Hidden size for MLP classifier (not needed to evaluate a model with XLNet architecture)")
	parser.add_argument("--drop_rate",
						default=0.3,
						type=float,
						help="Droprate in between hidden layers for MLP classifer (not needed to evaluate a model with XLNet architecture)")
	parser.add_argument("--activation_function",
						default='sigmoid',
						type=str,
						help="Activation function for MLP classifer (not needed to evaluate a model with XLNet architecture)")

	args = parser.parse_args()

	# Load training data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading {} dataset".format(args.set_type))
	test_dataloader = load_featurized_examples(batch_size=32, set_type = args.set_type, sliding_window = (args.model_type=="classifier"), feature_save_path=feature_save_path)

	# Load saved model
	model_path = os.path.join('/gpfs/data/razavianlab/capstone19/models/', args.model_id, 'model_checkpoint_'+args.checkpoint)
	logger.info("Loading saved model from {}".format(model_path))
	if args.model_type == "xlnet":
		config = XLNetConfig.from_pretrained(os.path.join(model_path, 'config.json'), num_labels=2292) # TODO: check if we need this
		model = XLNetForSequenceClassification.from_pretrained(model_path, config=config)
	else: 
		saved_model = torch.load(os.path.join(model_path, 'model.pt'))
		model = SlidingClassifier(num_layers=args.num_hidden_layers, hidden_size=args.hidden_size, p=args.drop_rate, activation_function=args.activation_function)
		model.state_dict = saved_model['model']
	model.to(device)
	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

	eval_folder = '/gpfs/data/razavianlab/capstone19/evals'
	val_file_name = os.path.join(eval_folder, args.model_id + "_{}_{}_metrics.p".format(args.checkpoint, args.set_type))
	# Create empty data frame to store evaluation results in (to be written to val_file_name)
	val_results = pd.DataFrame(columns=['loss', 'micro_AUC', 'macro_AUC', 'top1_precision', 'top3_precision', 'top5_precision', 'micro_f1', 'macro_f1', 'macro_AUC_list'])
	# Run evaluation
	results = evaluate(dataloader = test_dataloader, model = model, model_id = args.model_id, n_gpu=n_gpu, device=device, sliding_window = (args.model_type=="classifier"))
	# Save results
	val_results = val_results.append(pd.DataFrame(results, index=[0]))
	pickle.dump(val_results, open(val_file_name, "wb"))
	os.system("chgrp razavianlab {}".format(val_file_name))

	return

if __name__ == "__main__":
	main()
