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
	parser.add_argument("--save_batch",
						type=int,
						default=50000,
						help="The number of batches before saving a summarized set of sliding embeddings and labels")
	parser.add_argument("--set_type",
						type=str,
						help="Specify train/test file.")

	parser.add_argument("--batch_size",
                                                type=str,
                                                help="Specify batch size for load featurized examples.")
	
	args = parser.parse_args()

	# Load data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading test dataset")
	dataloader = load_featurized_examples(batch_size=args.batch_size, set_type = args.set_type, feature_save_path=feature_save_path)

	# Load saved model
	model_path = os.path.join('/gpfs/data/razavianlab/capstone19/models/', args.model_id, 'model_checkpoint_'+args.checkpoint)
	logger.info("Loading saved model from {}".format(model_path))
	config = XLNetConfig.from_pretrained(os.path.join(model_path, 'config.json'), num_labels=2292) # TODO: check if we need this
	model = XLNetForSequenceClassification.from_pretrained(model_path, config=config)
	model.to(device)
	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))


	last_batch_doc_id = -1 # Used to determine if the last document of the last batch was split up or not
	stored_logits = = torch.empty(0, 2292).to(device) # Stores logits until we finish a batch where the last document was not split up
	all_doc_ids = torch.empty(0).to(device) # Stores the list of doc ids corresponding to the rows of stored_logits
	all_combined_logits = torch.empty(0, 2292).to(device) # For all documents, stores the elementwise max of all logits for that document
	for i, batch in enumerate(dataloader):
		if i % 1000 == 0 and i > 0:
			logger.info('Entering batch {}'.format(i))
		model.eval()
		with torch.no_grad():
			
			input_ids, input_mask, segment_ids, label_ids, doc_ids = batch

			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()
			doc_ids = doc_ids.to(device).float()
			label_ids = label_ids.to(device).float()

			# Get logits for this batch 
			logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]

			# Check if any part of the last document in stored_logits is in this batch,
			# indicating that a document got split across stored_logits and this batch 
			if all(doc_ids != last_batch_doc_id) and  last_batch_doc_id != -1: # This means that the last batch of stored_logits did not get split up

				# If nothing was split, then we can combine the logits in stored_logits by document
				# and store the results in all_combined_logits

				# Combine logits by doc_id
				last_doc_id = all_doc_ids[0].item()
				to_combine = torch.empty(0, 2292).to(device) 
				for (j, doc_id) in enumerate(all_doc_ids):
					if doc_id.item() != last_doc_id: 
						# Get the pointwise max over all logits for the last document
						combined_logits = torch.max(to_combine, dim=0)[0] # pointwise max of all logits for the last document
						all_combined_logits = torch.cat([all_combined_logits, combined_logits], dim=0)
						# Create to_combine for the new document and update last_doc_id
						to_combine = stored_logits[j, :]
						last_doc_id = doc_id.item()
					else:
						# Add these logits to to_combine with the other logits for this document
						to_combine = torch.cat([to_combine, stored_logits[j, :]], dim=0)

				all_doc_ids = torch.empty(0).to(device)
				stored_logits = torch.empty(0, 2292).to(device)
				last_batch_doc_id =  doc_ids[-1]
			# If a doc was split, then save these logits until we find a batch where no doc was split
			else:
				stored_logits = np.concatenate((stored_logits, logits))
				all_doc_ids = torch.cat([all_doc_ids, doc_ids], dim = 0)
				last_batch_doc_id =  doc_ids[-1]

	preds = preds.append(torch.sigmoid(all_combined_logits).detach().cpu()) # sigmoid returns probabilities
	target = torch.load(os.path.join(feature_save_path, args.set_type + '_labels.pt'))
	preds = torch.cat(preds).numpy()
	target = torch.cat(target).byte().numpy()

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
