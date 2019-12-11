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
import random
from xlnet_evaluation import (macroAUC, topKPrecision)
import math

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
		input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
		input_mask = torch.load(os.path.join(feature_save_path, set_type + '_input_mask.pt'))
		segment_ids = torch.load(os.path.join(feature_save_path, set_type + '_segment_ids.pt'))
		labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
		doc_ids = torch.load(os.path.join(feature_save_path, set_type + '_doc_ids.pt'))
		
		data = TensorDataset(input_ids, input_mask, segment_ids, labels, doc_ids)

		sampler = SequentialSampler(data)

		dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = False)

		return dataloader
				

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

	parser.add_argument('--fp16',
											action='store_true',
											help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument("--feature_save_dir",
											type=str,
											help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/{feature_save_dir}'. ")
	parser.add_argument("--set_type",
											type=str,
											help="Specify train/val/test file.")
	parser.add_argument("--save_batch",
											type=int,
											help="Save files have been saved every save_batch batches.")
	parser.add_argument("--n_saved_batches",
											type=int,
											help="The saved files are split into n_saved_batches batches.")
	args = parser.parse_args()

	# Allocate space for preds and targets
	targets = np.zeros((args.save_batch*args.n_saved_batches, 2292))
	preds = np.zeros((args.save_batch*args.n_saved_batches, 2292))

	# Load data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	saved_so_far = 0
	for i in range(1, args.n_saved_batches + 1):
		if i%10 == 0 and i > 0:
			logger.info("Loading saved batch {}".format(i))
		batch_targets = torch.load(os.path.join(feature_save_path, "{}_label_ids_{}.pt".format(args.set_type, i)))
		batch_targets = torch.byte().detach().cpu().numpy()
		batch_preds = torch.load(os.path.join(feature_save_path, "{}_logits_{}.pt".format(args.set_type, i)))
		batch_preds = torch.sigmoid(batch_preds).detach().cpu()
		targets[saved_so_far:saved_so_far+ batch_targets.shape[0], :] = batch_targets
		preds[saved_so_far:saved_so_far + batch_preds.shape[0], :] = batch_preds
		saved_so_far = saved_so_far + batch_preds.shape[0]
	# Cut targets and preds to the right length (in case the final saved batched was shorter than args.save_batch)
	targets = targets[:saved_so_far, ]
	preds = preds[:saved_so_far, ]

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
	logger.info("micro_AUC : {} ,  macro_AUC : {}".format(str(micro_AUC) ,str(macro_AUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))
	logger.info("micro_f1 : {} , macro_f1 : {}".format(str(micro_f1), str(macro_f1)))


	return

if __name__ == "__main__":
	main()



