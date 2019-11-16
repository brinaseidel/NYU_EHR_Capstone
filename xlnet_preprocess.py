from transformers import XLNetTokenizer
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from scipy import stats
import torch
import os
import sys
import argparse

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

class DataProcessor(object):
	"""Converts data for sequence classification data sets."""

	def get_examples(self, filename, set_type, data_dir = '/gpfs/data/razavianlab/ehr_transformer/ICD_model/'):
		"""Gets a collection of `InputExample`s for the data set."""
		df = pd.read_csv(data_dir + filename)
		return self._create_examples(df, set_type)

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		return ["0", "1"]

	def _create_examples(self, df, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []

		for i in range(df.shape[0]):

			guid = "%s-%s" % (set_type, i)
			text_a = df['NOTE_TEXT'].iloc[i]
			label = df['ICD_DIAGNOSIS_CODE_cleaned'].iloc[i]

			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples

def loadICDDict(path, minCount = 1000):
	# create ICD dict, input is csv with ICD code and counts
	df = pd.read_csv(path)
	df = df[(pd.notnull(df['ICD_DIAGNOSIS_CODE'])) & (df['count'] >= minCount) & (df['ICD_DIAGNOSIS_CODE'] != '')].reset_index(drop = True)
	dictICD = dict(zip(df['ICD_DIAGNOSIS_CODE'], df.index))
	return dictICD

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id

def ICD2Idx(lsICD, dict):
	# Map icd code to index given dictionary, remove unknowns and duplicates
	lsIdx = [int(dict[x]) for x in lsICD if x in dict]
	cntUnk = len(lsICD) - len(lsIdx)
	lsIdx = list(set(lsIdx))
	return lsIdx, cntUnk

def convert_examples_to_features(examples, max_seq_length,
								 tokenizer, sliding_window=False):
	"""Loads a data file into a list of `InputBatch`s."""

	# Path to list of ICD codes and their count in the training data
	label_map = loadICDDict(path = '/gpfs/data/razavianlab/ehr_transformer/analysis/trainICD_count_afterFill.csv', minCount=1000)
	num_icd = len(label_map)
	logger.info("Number of ICD codes: {}".format(num_icd))

	unk_token = tokenizer.unk_token
	unk_id = tokenizer.convert_tokens_to_ids(unk_token)

	def add_special_tokens(tokens):

		# Add tokens for start and end of sequence
		tokens = ["<cls>"] + tokens + ["<sep>"]
		segment_ids = [0] * len(tokens)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		# The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		return input_ids, input_mask, segment_ids

	features = []
	for (doc_id, example) in enumerate(examples):
		if doc_id % 10000 == 0:
			logger.info("Writing example %d of %d" % (doc_id, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)

		if sliding_window == False:

			# Truncate tokens to max_seq_length, accounting for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

			# Add special tokens and create ids, mask, and segment ids
			input_ids, input_mask, segment_ids = add_special_tokens(tokens_a)

			# Pre-process labels
			icd_id, unk = ICD2Idx(example.label.split() , label_map)
			icd_id = [i for i in icd_id if i < num_icd]
			binary_label = np.zeros(num_icd)
			for l in icd_id:
				binary_label[l] = 1

			# Add this example to features
			features.append(
				InputFeatures(input_ids = torch.tensor(input_ids, dtype=torch.short),
							  input_mask = torch.tensor(input_mask, dtype=torch.int8),
							  segment_ids = torch.tensor(segment_ids, dtype=torch.int8),
							  label_id= torch.tensor(binary_label, dtype=torch.int8)))

		else:

			# Split tokens into windows of size max_seq_length, accounting for [CLS] and [SEP] with "- 2"
			n_splits = np.ceil(len(tokens_a)/(max_seq_length-2))
			print("n_splits: ", n_splits)
			tokens_a_list = []
			for i in range(n_splits-1):
				tokens_a_list.append(tokens_a[i*(max_seq_length - 2):(i+1)*(max_seq_length - 2)]) 
			tokens_a_list.append(tokens_a[i*(max_seq_length - 2):])

			# Add special tokens and create ids, mask, and segment ids
			input_ids_list = []
			input_mask_list = []
			segment_ids_list = []
			doc_id_list = []
			for i in range(n_splits):
				input_ids, input_mask, segment_ids = add_special_tokens(tokens_a_list[i])
				input_ids_list.append(input_ids)
				input_mask_list.append(input_mask)
				segment_ids_list.append(segment_ids)
				doc_id_list.append(doc_id)

			# Pre-process labels
			icd_id, unk = ICD2Idx(example.label.split() , label_map)
			icd_id = [i for i in icd_id if i < num_icd]
			binary_label = np.zeros(num_icd)
			for l in icd_id:
				binary_label[l] = 1
			binary_label_list = [binary_label for i in range(n_splits)]

			# Add all examples for this document to features
			for i in range(n_splits):
				features.append(
					InputFeatures(input_ids = torch.tensor(input_ids_list[i], dtype=torch.short),
								  input_mask = torch.tensor(input_mask_list[i], dtype=torch.int8),
								  segment_ids = torch.tensor(segment_ids_list[i], dtype=torch.int8),
								  label_id= torch.tensor(binary_label_list[i], dtype=torch.int8),
								  doc_id= torch.tensor(doc_id_list[i], dtype=torch.int8)))


	return features


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--filename",
						default="",
						type=str,
						help="Cleaned data file to be converted into features")

	parser.add_argument("--set_type",
						type=str,
						help="Specify train/val/test")

	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="Maximum length of input sequence")
	parser.add_argument("--feature_save_dir",
						type=str,
						help="Preprocessed data (features) will be saved at '/gpfs/data/razavianlab/capstone19/preprocessed_data/feature_save_dir'. ")

	parser.add_argument("--sliding_window",
						default=False,
						type=bool,
						help="True/False determining whether to preprocess the data in sliding windows")

	args = parser.parse_args()

	# Section: Set device for PyTorch
	if torch.cuda.is_available():
		 # might need to update when using more than 1 GPU
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	processor = DataProcessor()

	#Load training data
	examples = processor.get_examples(filename = args.filename , set_type = args.set_type)

	tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
	
	logger.info("***** Converting examples to features *****")
	features = convert_examples_to_features(examples, max_seq_length = args.max_seq_length, tokenizer=tokenizer, sliding_window=args.sliding_window)

	logger.info("  Num examples = %d", len(examples))

	feature_save_path = os.path.join('/gpfs/data/razavianlab/capstone19/preprocessed_data/', args.feature_save_dir)
	if not os.path.exists(feature_save_path):
		os.makedirs(feature_save_path)

	all_input_ids = torch.stack([f.input_ids for f in features])
	all_input_mask = torch.stack([f.input_mask for f in features])
	all_segment_ids = torch.stack([f.segment_ids for f in features])
	all_label_ids = torch.stack([f.label_id for f in features])

	torch.save(all_input_ids , os.path.join(feature_save_path, args.set_type + '_input_ids.pt'))
	torch.save(all_input_mask , os.path.join(feature_save_path, args.set_type + '_input_mask.pt'))
	torch.save(all_segment_ids , os.path.join(feature_save_path, args.set_type + '_segment_ids.pt'))
	torch.save(all_label_ids , os.path.join(feature_save_path, args.set_type + '_labels.pt'))

if __name__ == "__main__":
	main()
