import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union
import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation")

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score)

## Modules written from scratch
from modules.utils import setup_logging, log_resources, create_dataloaders, preprocess_data
from modules.networks import ResNet
from modules.datasets import ECGDataset
from modules.data_preprocessor import ECGDataPreprocessor
from modules.visualization import (plot_and_save_individual_confusion_matrices, plot_metrics, plot_roc_curve,
                         plot_precision_recall_curve, plot_roc_curve_per_class, plot_precision_recall_curve_per_class)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Arguments")

    # Data path
    parser.add_argument('--data_path', type=str, default='../../data/ptbxl/', help='Path to the data directory')

    # Model config
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[7, 5, 3], help='List of kernel sizes for convolution layers')
    parser.add_argument('--blocks_channels', type=int, nargs='+', default=[64, 128, 256], help='Size of channels for ResBlocks')
    parser.add_argument('--blocks_layers', type=int, nargs='+', default=[2, 2, 2], help='Layers per ResBlock')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Sampling rate of the data')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    # parser.add_argument('--val_every', type=int, default=2, help='Validate every N epochs') ## Validate every epoch: TODO
    parser.add_argument('--shuffle', dest='shuffle', action='store', type=str, default='true',
                    choices=['true', 'false'], help="Shuffle training data -> default True")
    parser.add_argument('--normalization', type=str, choices=['standardize', 'minmax'], default='standardize', help="Type of normalization to apply (standardize or minmax).")
    args = parser.parse_args()
    args.shuffle = args.shuffle.lower() == 'true'


    # Setup directory paths
    date = datetime.now().strftime('%Y_%m_%d')
    time = datetime.now().strftime('%H_%M_%S')
    experiment_dir = Path(f"experiments/{date}/{time}")
    results_dir = experiment_dir.joinpath("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = "log_training_pipeline.log"
    logger = setup_logging(log_file, experiment_dir)

    logger.info('Training parameters:')
    # Log the parsed arguments
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Data shuffle: {args.shuffle}")
    logger.info(f"Normalization: {args.normalization}")
    logger.info(f"Kernel size: {args.kernel_size}")
    logger.info(f"Blocks channels: {args.blocks_channels}")
    logger.info(f"Blocks layers: {args.blocks_layers}")

    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Sampling rate: {args.sampling_rate}")
    logger.info(f"Patience: {args.patience}\n")
    # logger.info(f"Validate every: {args.val_every}") --> TODO

    logger.info("Loading data...") 
    ecg_dataset = ECGDataset(data_dir=Path(args.data_path), sample_rate=args.sampling_rate)
    raw_data = ecg_dataset.get_raw_data()
    logger.info("Data successfully loaded!\n")

    logger.info("Preprocessing data...") 
    preprocessors, data_info = preprocess_data(
        raw_data=raw_data,
        sample_rate=args.sampling_rate,
        shuffle= args.shuffle,
        normalization_type = args.normalization,
        segment_length = 0,
        apply_denoising = False,
        apply_augmentation=False,
    )

    logger.info("Data successfully processed!\n")

    logger.info("Splitting data...") 
    dataloaders = create_dataloaders(preprocessors, batch_size=args.batch_size)
    logger.info("Train, val and test splitted successfullly!\n")
    logger.info("Initializing model: ")
    num_classes = data_info['num_classes']
    logger.info(f"Num classes: {num_classes}")
    channels = data_info['channels']
    logger.info(f"Channels: {channels}")
    labels_class = data_info['labels_class']
    logger.info(f"Labels class: {labels_class}")

    #Load the model
    model = ResNet(num_classes=num_classes, input_channels= channels, block_layers= args.blocks_layers, block_channels= args.blocks_channels, kernel_sizes=args.kernel_size)
    # Move model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model successfully init!\n")

    logger.info("Hyperparams sestup:")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                        max_lr=args.learning_rate,
                                                        steps_per_epoch=len(dataloaders['train']),
                                                        epochs=args.epochs)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Early stopping parameters
    save_path= experiment_dir.joinpath("model_checkpoint.pth")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_steps = args.patience
    # Initialize history dictionary
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    logger.info("Hyperparams initialized!\n")

    # Start training
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for data, target in dataloaders['train']:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()         
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(dataloaders['train'].dataset)

        # Validation loop
        model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for data, target in dataloaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item()
                # Store predictions and targets
                all_targets.append(target.cpu())
                all_predictions.append(torch.sigmoid(output).cpu()) 

        val_loss /= len(dataloaders['val'].dataset)
        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)

        # Compute metrics
        predictions = (all_predictions > 0.5).float()
        accuracy = accuracy_score(all_targets.numpy(), predictions.numpy())
        precision = precision_score(all_targets.numpy(), predictions.numpy(), average='macro')
        recall = recall_score(all_targets.numpy(), predictions.numpy(), average='macro')
        f1 = f1_score(all_targets.numpy(), predictions.numpy(), average='macro')

        # Store the metrics
        current_lr = scheduler.get_last_lr()[0]
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)

        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        logger.info(f"Validation: Accuracy: {accuracy:.4f}, Precision: {precision:.4f},  Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the best model and training state
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, save_path)
            logger.info(f"Model checkpoint saved to {save_path}")
        else:
            early_stopping_counter += 1
            logger.info(f"\tEarly stopping counter {early_stopping_counter}/{early_stopping_steps}")
        if early_stopping_counter >= early_stopping_steps:
            logger.info(f"\tEarly stopping triggered at epoch {epoch + 1}")
            break

    # Load the best model for evaluation
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save training history
    torch.save(history, experiment_dir.joinpath('training_history.pth'))
    logger.info("Training history saved to training_history.pth")
    plot_metrics(history, results_dir, 'val')
    logger.info("Training completed!\n")

    ## EVALUATE ON TEST SET ##
    logger.info("Starting evaluation on test set:")
    # Compute classification report
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in dataloaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_targets.append(target.cpu())
            all_predictions.append(torch.sigmoid(output).cpu())  # Assuming binary classification

    all_targets = torch.cat(all_targets)
    all_predictions = torch.cat(all_predictions)
    predictions = (all_predictions > 0.5).float()

    class_report = classification_report(all_targets.numpy(), predictions.numpy(), target_names=labels_class, zero_division=0)
    logger.info(f"Classification Report:\n{class_report}")

    # Optionally, you can save or plot the classification report
    with open(results_dir.joinpath('test_classification_report.txt'), 'w') as f:
        f.write(class_report)
    logger.info("Classification report saved to classification_report.txt")

    # Compute and log metrics
    accuracy = accuracy_score(all_targets.numpy(), predictions.numpy())
    precision = precision_score(all_targets.numpy(), predictions.numpy(), average='macro')
    recall = recall_score(all_targets.numpy(), predictions.numpy(), average='macro')
    f1 = f1_score(all_targets.numpy(), predictions.numpy(), average='macro')

    logger.info(f"Test Accuracy: {accuracy:.3f}")
    logger.info(f"Test Precision (Macro): {precision:.3f}")
    logger.info(f"Test Recall (Macro): {recall:.3f}")
    logger.info(f"Test F1-Score (Macro): {f1:.3f}\n")

    logger.info("Plotting results..")
    plot_and_save_individual_confusion_matrices(true_labels=all_targets.numpy(),predicted_labels=predictions.numpy(), class_names=labels_class,save_dir=results_dir, data_type = 'test')
    plot_roc_curve(all_targets, all_predictions, results_dir, 'test')
    plot_precision_recall_curve(all_targets, all_predictions, results_dir, 'test')
    plot_roc_curve_per_class(all_targets, all_predictions, labels_class, results_dir, 'test')
    plot_precision_recall_curve_per_class(all_targets, all_predictions, labels_class, results_dir, 'test')
    logger.info("Sucessfully plotted!\n")

    logger.info("Script compiled successfully. Closing.")
