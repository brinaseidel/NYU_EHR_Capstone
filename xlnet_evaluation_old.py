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

	micro_AUC = metrics.roc_auc_score(target, preds, average='micro')
	macro_AUC, macro_AUC_list = macroAUC(preds, target)
	top1_precision = topKPrecision(preds, target, k = 1)
	top3_precision = topKPrecision(preds, target, k = 3)
	top5_precision = topKPrecision(preds, target, k = 5)
	f1 = metrics.f1_score(target, preds)

	logger.info("Evaluation loss : {}".format(str(eval_loss)))
	logger.info("micro_AUC : {} ,  macro_AUC : {}".format(str(micro_AUC) ,str(macro_AUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))
	logger.info("F1 : {}".format(str(f1)))

	results = {
				'loss': eval_loss,
				'micro_AUC' : micro_AUC ,
				'macro_AUC' : macro_AUC,
				'top1_precision' : top1_precision ,
				'top3_precision' : top3_precision ,
				'top5_precision' : top5_precision,
				'f1' : f1,
				'macro_AUC_list' : macro_AUC_list
				}

	return results





















def main():

	# TODO: Add all main methods
	return

if __name__ == "__main__":
	main()
