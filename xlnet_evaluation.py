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

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
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
	return np.mean(auc), auc

def topKPrecision(pred, true, k):
	# pred: size of n_sample x n_class
	idx_sort = np.argsort(pred, axis=1)
	n_sample = true.shape[0]
	idx_row = np.array([int(x) for x in range(n_sample)]).reshape(-1, 1)
	true_sort = true[idx_row, idx_sort]
	result = np.sum(true_sort[:,-k:].astype(np.float64)) / k / n_sample
	return result

def evaluate(dataloader, model, model_id, n_gpu, device):
	logger.info("***** Running evaluation *****")
	logger.info("  Num batches = %d", len(dataloader))
	eval_loss = 0.0
	number_steps = 0
	preds = []
	target = []
	
	for batch in dataloader:
		model.eval()
		input_ids, input_mask, segment_ids, label_ids = batch
		input_ids = input_ids.to(device).long()
		input_mask = input_mask.to(device).long()
		segment_ids = segment_ids.to(device).long()
		label_ids = label_ids.to(device).float()
		
		with torch.no_grad():
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

	logger.info("Evaluation loss : {}".format(str(eval_loss)))
	logger.info("micro_AUC : {} ,  macro_AUC : {}".format(str(micro_AUC) ,str(macro_AUC)))
	logger.info("top1_precision : {} ,  top3_precision : {}, top5_precision : {}".format(str(top1_precision), str(top3_precision), str(top5_precision)))

	results = { 
				'loss': eval_loss,  
				'micro_AUC' : micro_AUC ,
				'macro_AUC' : macro_AUC, 
				'top1_precision' : top1_precision , 
				'top3_precision' : top3_precision , 
				'top5_precision' : top5_precision,
				'macro_AUC_list': macro_AUC_list
				}

	return results


















		
		

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
	args = parser.parse_args()

	# Load training data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading test dataset")
	test_dataloader = load_featurized_examples(batch_size=32, set_type = "test", feature_save_path=feature_save_path)

	# Load saved model
	model_path = os.path.join('/gpfs/data/razavianlab/capstone19/models/', args.model_id, 'model_checkpoint_'+args.checkpoint)
	logger.info("Loading saved model from {}".format(model_path))
	config = XLNetConfig.from_pretrained(os.path.join(model_path, 'config.json'), num_labels=2292) # TODO: check if we need this
	model = XLNetForSequenceClassification.from_pretrained(model_path, config=config)
	model.to(device)
	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

	eval_folder = '/gpfs/data/razavianlab/capstone19/evals'
	val_file_name = os.path.join(eval_folder, model_id + "_test_metrics.p")
	# Run evaluation
	# Create empty data frame to store evaluation results in (to be written to val_file_name)
    val_results = pd.DataFrame(columns=['loss', 'micro_AUC', 'macro_AUC', 'top1_precision', 'top3_precision', 'top5_precision', 'macro_AUC_list'])
	results = evaluate(dataloader = test_dataloader, model = model, model_id = args.model_id, n_gpu=n_gpu, device=device)
	val_results = val_results.append(pd.DataFrame(results, index=[global_step]))
	pickle.dump(val_results, open(val_file_name, "wb"))
	os.system("chgrp razavianlab {}".format(val_file_name))


	return

if __name__ == "__main__":
	main()
