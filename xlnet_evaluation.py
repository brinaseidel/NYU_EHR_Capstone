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
from sklearn.metrics import matthews_corrcoef, f1_score ,roc_auc_score

def macroAUC(pred, true):
	auc = []
	for i in range(pred.shape[1]):
		if (len(np.unique(true[:,i])) > 1) :
			auc.append(metrics.roc_auc_score(true[:,i], pred[:,i]))
		else:
			auc.append(0.5)
	return np.mean(auc)

def topKPrecision(pred, true, k):
	# pred: size of n_sample x n_class
	idx_sort = np.argsort(pred, axis=1)
	n_sample = true.shape[0]
	idx_row = np.array([int(x) for x in range(n_sample)]).reshape(-1, 1)
	true_sort = true[idx_row, idx_sort]
	result = np.sum(true_sort[:,-k:].astype(np.float64)) / k / n_sample
	return result

def evaluate(dataloader, model,eval_file_name, eval_folder = '/gpfs/data/razavianlab/capstone19/evals'):
	logger.info("***** Running evaluation *****")
	logger.info("  Num batches = %d", len(dataloader))
	eval_loss = 0.0
	number_steps = 0
	preds = []
	target = []
	
	with tqdm(total=len(dataloader), desc=f"Evaluating") as progressbar:
		for batch in dataloader:
			model.eval()
			input_ids, input_mask, segment_ids, label_ids = batch
			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()

			# Might need to add .half() or .long() depending on amp versions
			label_ids = label_ids.to(device)
			with torch.no_grad():
				logits = model(input_ids, segment_ids, input_mask, labels=None)
			criterion = BCEWithLogitsLoss()
			loss = criterion(logits, label_ids)

			# TODO: Check why we take mean
			print("loss = ", loss)
			if n_gpu > 1:
				eval_loss += loss.mean().item()
			else:
				eval_loss += loss.item()

			number_steps += 1
			preds.append( logits.detach().cpu())
			target.append(label_ids.detach().cpu())
			progressbar.update(1)
			# not used in calculations, for sanity checks
			mean_loss = eval_loss/number_steps
			progressbar.set_postfix_str(f"Loss: {mean_loss:.5f}")

	eval_loss = eval_loss / number_steps
	preds = torch.cat(preds).numpy()
	target = torch.cat(target).byte().numpy()

	microAUC = roc_auc_score(target, preds, average='micro')
	macroAUC = macroAUC(preds, target)
	top1_precision = topKPrecision(preds, target, k = 1)
	top3_precision = topKPrecision(preds, target, k = 3)
	top5_precision = topKPrecision(preds, target, k = 5)

	logger.info("microAUC : {} ,  macroAUC : {}".format(str(microAUC) ,str(macroAUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))

	results = { 
				'loss': eval_loss,  
				'microAUC' : microAUC ,
				'macroAUC' : macroAUC, 
				'top1_precision' : top1_precision , 
				'top3_precision' : top3_precision , 
				'top5_precision' : top5_precision
				}

	eval_save_path = os.path.join(eval_folder, eval_file_name)
	with open(eval_save_path, 'w') as outfile:
		json.dump(results, outfile)


















		
		

def main():

	# TODO: Add all main methods 


if __name__ == "__main__":
	main()
