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
import math

def load_featurized_examples(batch_size, set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/'):
	input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
	input_mask = torch.load(os.path.join(feature_save_path, set_type + '_input_mask.pt'))
	segment_ids = torch.load(os.path.join(feature_save_path, set_type + '_segment_ids.pt'))
	labels = torch.load(os.path.join(feature_save_path, set_type + '_labels.pt'))
	data = TensorDataset(input_ids, input_mask, segment_ids, labels)

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

	parser.add_argument("--feature_save_dir",
						type=str,
						help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/{feature_save_dir}'. ")

	parser.add_argument("--set_type",
						type=str,
						help="Specify train/test file.")

	args = parser.parse_args()

	# Load training data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading {} dataset".format(args.set_type))
	dataloader = load_featurized_examples(batch_size=32, set_type = args.set_type, feature_save_path=feature_save_path)

	# Load saved model
	config = XLNetConfig.from_pretrained('xlnet-base-cased', num_labels=2292)
	model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', config=config)
	model.to(device)
	model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

	summaries = torch.empty(0, config.d_model).to(device)
	labels= torch.empty(0, config.num_labels).to(device)
	for i, batch in enumerate(dataloader):
		model.eval()
		with torch.no_grad():
			input_ids, input_mask, segment_ids, label_ids = batch

			input_ids = input_ids.to(device).long()
			input_mask = input_mask.to(device).long()
			segment_ids = segment_ids.to(device).long()
			label_ids = label_ids.to(device).float()
			transformer_outputs = model.module.transformer(input_ids = input_ids,
													token_type_ids=segment_ids,
													input_mask=input_mask)

			output = transformer_outputs[0]
			# extracting the CLS token
			summary = output[:,0]
			summary = summary.to(device)

			summaries = torch.cat([summaries, summary], dim = 0)
			labels = torch.cat([labels, label_ids])
			
			if i%1000 == 0 and i > 0:
				logger.info("Embedded and summarized batch {} of {}".format(i, len(dataloader)))

		# Save the embedded representations of the document every 50,000 batches to save memory
		if i%12000 == 0 and i >0:
			logger.info("Saving summaries...")
			torch.save(summaries, os.path.join(feature_save_path, args.set_type + '_summaries_{}.pt'.format(int(i/12000))))
			torch.save(labels, os.path.join(feature_save_path, args.set_type + '_label_ids_{}.pt'.format(int(i/12000))))
			summaries = torch.empty(0, config.d_model).to(device)
			labels= torch.empty(0, config.num_labels).to(device)

	# Save any remaining embedded representations
	if i%12000 != 0:
		logger.info("Saving summaries...")
		torch.save(summaries, os.path.join(feature_save_path, args.set_type + '_summaries_{}.pt'.format(int(math.ceil(i/12000)))))
		torch.save(labels, os.path.join(feature_save_path, args.set_type + '_label_ids_{}.pt'.format(int(math.ceil(i/12000)))))
	
	return

if __name__ == "__main__":
	main()
