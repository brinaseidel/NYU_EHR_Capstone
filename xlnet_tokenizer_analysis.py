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
from transformers import XLNetTokenizer
import random
import string
import pickle
import sentencepiece as spm

def load_featurized_examples(set_type, feature_save_path = '/gpfs/data/razavianlab/capstone19/preprocessed_data/small/', batch_size=1):
	input_ids = torch.load(os.path.join(feature_save_path, set_type + '_input_ids.pt'))
	data = TensorDataset(input_ids)

	# Note: Possible to use SequentialSampler for eval, run time might be better
	sampler = RandomSampler(data)

	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last = True)

	return dataloader

# TODO: Update to work in the tokenizezr analysis (this is originally from xlnet_preprocess.py)
def calculate_oov_statistics(count_unknowns, lengths, max_seq_length):
	included_tokens = [min(length, max_seq_length) for length in lengths]
	per_example_oov = [count_unknowns[i]/included_tokens[i] for i in range(len(included_tokens))]
	logger.info(stats.describe(np.array(per_example_oov)))
	return per_example_oov


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

	parser.add_argument("--set_type",
					type=str,
					help="Specify train/val/test")
	parser.add_argument("--feature_save_dir",
					type=str,
					help="Preprocessed data (features) should be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/feature_save_dir'. ")
	parser.add_argument("--sample_size",
					default=10000000,
					type=int,
					help="Sample size if taking a sample for statistical calculation (otherwise use the full dataset)")
	args = parser.parse_args()

	# Load data
	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	logger.info("Loading {} dataset".format(args.set_type))
	dataloader = load_featurized_examples(set_type = args.set_type, feature_save_path=feature_save_path)
	logger.info("Finished loading dataset")

	# Create XLNET pretrained tokenizer
	tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
	vocab_file_directory = os.path.join('/gpfs/data/razavianlab/capstone19/tokenizer_analysis/', args.feature_save_dir)
	tokenizer.save_vocabulary(vocab_file_directory)
	sp = spm.SentencePieceProcessor()
	sp.Load(os.path.join(vocab_file_directory, "spiece.model"))
	vocab_list = [sp.IdToPiece(id) for id in range(sp.GetPieceSize())]

	logger.info("Loaded in vocab")
	print("Unk token: {}".format(tokenizer.unk_token_id))
	print("Cls token: {}".format(tokenizer.cls_token_id))
	print("Sep token: {}".format(tokenizer.sep_token_id))
	strings_to_remove = string.punctuation
	ids_to_remove = []
	for s in strings_to_remove:
		ids_to_remove.append(tokenizer.convert_tokens_to_ids(s))

	special_token_ids = [tokenizer.unk_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
	ids_to_remove = ids_to_remove + special_token_ids

	weird_tokens = ['',' ', '<']
	splitting_rates = []
	individually_decoded_lengths = []
	tokens_split_on_spaces_punct_lengths = []
	avg_in_vocab_percentage = []
	total_in_vocab_percentage = []

	logger.info("Starting calculations")
	for i, batch in enumerate(dataloader):
		if args.sample_size > i:
			input_ids = batch[0]

			input_ids = input_ids.tolist()[0]
			input_ids = [token_id for token_id in input_ids if (token_id not in special_token_ids)]

			# Remove special tokens for analysis
			clean_input_ids = [token_id for token_id in input_ids if (token_id not in ids_to_remove)]

			# Reconstruct note from tokenizer and ids
			note = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)

			# Create simple split on whitespace and punctuation for comparison
			tokens_split_on_spaces_punct = note.translate(str.maketrans('','',string.punctuation)).split()

			in_vocab = [1 for token in tokens_split_on_spaces_punct if token in vocab_list]

			individually_decoded = []
			for input_id in clean_input_ids:
				individually_decoded.append(tokenizer.decode([input_id], clean_up_tokenization_spaces=False))

			individually_decoded = [token_id for token_id in individually_decoded if (token_id not in weird_tokens)]
			individually_decoded_lengths.append(len(individually_decoded))
			tokens_split_on_spaces_punct_lengths.append(len(tokens_split_on_spaces_punct))
			splitting_rates.append(len(individually_decoded)/len(tokens_split_on_spaces_punct))
			avg_in_vocab_percentage.append(sum(in_vocab)/len(tokens_split_on_spaces_punct))
			total_in_vocab_percentage.append(sum(in_vocab))

			# Print first 5
			if i < 5:
				print("Clean input IDs:\n{}\n".format(clean_input_ids))
				print("Reconstructed text of the note:\n{}\n".format(note))
				print("Tokens split on spaces and punctuation:\n{}\n".format(tokens_split_on_spaces_punct))
				print("Individually decoded clean input ids:\n {}\n".format(individually_decoded))

			if i % 100000 == 0:
				logger.info("Analyzed {} examples".format(i))

	print("Average splitting rate: {}".format(np.mean(splitting_rates)))
	print("Average in-vocab percentage: {}".format(np.mean(avg_in_vocab_percentage)))
	print("Total in-vocab percentage: {}".format(sum(total_in_vocab_percentage)/sum(tokens_split_on_spaces_punct_lengths)))
	save_file_name = os.path.join('/gpfs/data/razavianlab/capstone19/tokenizer_analysis/', args.feature_save_dir + '_vocab_tokenizer_lists.p')
	saved_lists = {'splitting_rates':splitting_rates, 'individually_decoded_lengths':individually_decoded_lengths, 'tokens_split_on_spaces_punct_lengths': tokens_split_on_spaces_punct_lengths, 'avg_in_vocab_percentage': avg_in_vocab_percentage, 'total_in_vocab_percentage': total_in_vocab_percentage}
	pickle.dump(saved_lists, open(save_file_name, "wb"))

if __name__ == "__main__":
	main()

