import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union
import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation")

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss, jaccard_score,
                             precision_score, recall_score)

from data import dataset, dataloader
from evaluation import visualize as mlplots
from utils.utilities import EarlyStopping, balance_input
from utils.logger import setup_logging, log_resources
from models.architectures import FullyConvolutionalNetwork
from models.pipeline import Pipeline

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Arguments")

    # Data path
    parser.add_argument('--data_path', type=str, default='../data/ptbxl/', help='Path to the data directory')

    # Model config
    parser.add_argument('--filters', type=int, nargs='+', default=[128, 256, 128], help='List of filters for convolution layers')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[8, 5, 3], help='List of kernel sizes for convolution layers')
    parser.add_argument('--linear_layer_len', type=int, default=128, help='Size of the linear layer')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Sampling rate of the data')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--delta_stop', type=float, default=0, help='Minimum change to qualify as an improvement')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision for training')

    args = parser.parse_args()

    logger = setup_logging("training_pipeline.log")

    logger.info('Training parameters:')
    # Log the parsed arguments
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Filters: {args.filters}")
    logger.info(f"Kernel size: {args.kernel_size}")
    logger.info(f"Linear layer length: {args.linear_layer_len}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Sampling rate: {args.sampling_rate}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Delta stop: {args.delta_stop}")
    logger.info(f"Validate every: {args.val_every}")
    logger.info(f"Half precision: {args.half_precision}\n")


    logger.info("Loading data...")
    data = dataloader.load_data(
        data_path=args.data_path,  
        sampling_rate=args.sampling_rate)  
    logger.info("Data loaded.")

    for aug_type in [None, "under", "over"]:
        train_ds = dataset.ECGDataset(data=data, apply_sampler=False, ds_type="train", shuffle=True, augmentation=aug_type)
        val_ds = dataset.ECGDataset(data=data, apply_sampler=False, ds_type="val", shuffle=True)
        test_ds = dataset.ECGDataset(data=data, apply_sampler=False, ds_type="test")

        model = FullyConvolutionalNetwork(num_classes=train_ds.num_classes,
                                          channels=train_ds.channels,
                                          filters=args.filters,  
                                          kernel_sizes=args.kernel_size, 
                                          linear_layer_len=args.linear_layer_len) 

        pipe = Pipeline(model=model, labels_name=train_ds.labels_class)

        pipe.train(train_ds=train_ds,
                   val_ds=val_ds,
                   epochs=args.epochs,  
                   batch_size=args.batch_size,  
                   lr=args.learning_rate,  
                   patience=args.patience,  
                   delta_stop=args.delta_stop,  
                   val_every=args.val_every,  
                   half_precision=args.half_precision)

        pipe.evaluate(test_ds = test_ds, batch_size = args.batch_size)
    
    logger.info("Script compiled successfully. Closing.")