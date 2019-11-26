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

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
	input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
	input_mask = torch.load(os.path.join(feature_save_path, set_type + '_input_mask.pt'))
	segment_ids = torch.load(os.path.join(feature_save_path, set_type + '_segment_ids.pt'))
	labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
	doc_ids = torch.load(os.path.join(feature_save_path, set_type + '_doc_ids.pt'))
	data = TensorDataset(input_ids, input_mask, segment_ids, labels, doc_ids)

	sampler = SequentialSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = True)

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

	parser.add_argument("--set_type",
						type=str,
						help="Specify train/test file.")

	args = parser.parse_args()

	# Load training data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading test dataset")
	test_dataloader = load_featurized_examples(batch_size=32, set_type = args.set_type, feature_save_path=feature_save_path)

	# Load saved model
	model_path = os.path.join('/gpfs/data/razavianlab/capstone19/models/', args.model_id, 'model_checkpoint_'+args.checkpoint)
	logger.info("Loading saved model from {}".format(model_path))
	config = XLNetConfig.from_pretrained(os.path.join(model_path, 'config.json'), num_labels=2292) # TODO: check if we need this
	model = XLNetForSequenceClassification.from_pretrained(model_path, config=config)
	model.to(device)
	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

	summaries = torch.empty(0, config.d_model).to(device)
	all_doc_ids = torch.empty(0).to(device)
	all_label_ids = torch.empty(0, 2292).to(device)
	try_to_save = False
	num_saved = 1
	for i, batch in enumerate(test_dataloader):
		if i % 10 == 0 and i > 0:
			logger.info("Processing record {}".format(i))
			try_to_save = True
		model.eval()
		with torch.no_grad():
			input_ids, input_mask, segment_ids, label_ids, doc_ids = batch

			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()
			doc_ids = doc_ids.to(device).float()
			label_ids = label_ids.to(device).float()
			# Try to save periodically
			if try_to_save: 
				# Check if the docs got split across batches, where last_batch_doc_id would be equal to any of the current doc_ids
				if all(doc_ids != last_batch_doc_id):
					# Average the representation of the CLS token for all examples from the same document
					mask = torch.zeros(int(all_doc_ids.max().item())+1, len(summaries))
					mask[all_doc_ids.long(), torch.arange(len(summaries))] = 1
					averaging_matrix = torch.nn.functional.normalize(mask, p=1, dim=1).to(device)
					mean_summaries = torch.mm(averaging_matrix, summaries)

					# Create an object storing one copy of the labels per document
					last_doc_id = -1
					label_ids = torch.empty(0, all_label_ids.size()[1]).to(device)
					for (j, doc_id) in enumerate(all_doc_ids):
						if doc_id.item() != last_doc_id:
							label_ids = torch.cat([label_ids, all_label_ids[j].unsqueeze(0)])
							last_doc_id = doc_id.item()
					# Save the embedded representations of the document, along with the labels
					# print('Last doc id',last_batch_doc_id)
					# print('Doc ids',doc_ids)
					torch.save(mean_summaries, os.path.join(feature_save_path, args.set_type + '_summaries_{}.pt'.format(num_saved)))
					torch.save(label_ids, os.path.join(feature_save_path, args.set_type + '_doc_label_ids_{}.pt'.format(num_saved)))
					num_saved += 1
					summaries = torch.empty(0, config.d_model).to(device)
					all_doc_ids = torch.empty(0).to(device)
					all_label_ids = torch.empty(0, 2292).to(device)
					try_to_save = False
			transformer_outputs = model.module.transformer(input_ids = input_ids, 
													token_type_ids=segment_ids,
													input_mask=input_mask)
			
			output = transformer_outputs[0]
			# extracting the CLS token
			summary = output[:,0]
			summary = summary.to(device)

			summaries = torch.cat([summaries, summary], dim = 0)
			all_doc_ids = torch.cat([all_doc_ids, doc_ids], dim = 0)
			all_label_ids = torch.cat([all_label_ids, label_ids], dim = 0)
			last_batch_doc_id =  doc_ids[-1]

	# Average the representation of the CLS token for all examples from the same document
	mask = torch.zeros(int(all_doc_ids.max().item())+1, len(summaries))
	mask[all_doc_ids.long(), torch.arange(len(summaries))] = 1
	averaging_matrix = torch.nn.functional.normalize(mask, p=1, dim=1).to(device)
	mean_summaries = torch.mm(averaging_matrix, summaries)

	# Create an object storing one copy of the labels per document
	last_doc_id = -1
	label_ids = torch.empty(0, all_label_ids.size()[1]).to(device)
	for (j, doc_id) in enumerate(all_doc_ids):
		if doc_id.item() != last_doc_id:
			label_ids = torch.cat([label_ids, all_label_ids[j].unsqueeze(0)])
			last_doc_id = doc_id.item()

	# Save the embedded representations of the document, along with the labels
	torch.save(mean_summaries, os.path.join(feature_save_path, args.set_type + '_summaries_{}.pt'.format(num_saved+1)))
	torch.save(label_ids, os.path.join(feature_save_path, args.set_type + '_doc_label_ids.pt'.format(num_saved+1))) # label_ids.pt has one record per window (and thus multiple records per document)	

	return

if __name__ == "__main__":
	main()
