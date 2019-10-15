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

def evaluate(dataloader, model, eval_save_path):
    logger.info("***** Running evaluation *****")
    logger.info("  Num batches = %d", len(dataloader))
    eval_loss = 0.0
    preds = None
    out_label_ids = None
    for batch in tqdm(dataloader, desc="Evaluating"):
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
        eval_loss += loss.mean().item()

		# TODO: Finish evaluation function


def main():

	# TODO: Add all main methods 


if __name__ == "__main__":
	main()
