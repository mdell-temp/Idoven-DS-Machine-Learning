import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss, jaccard_score,
                             precision_score, recall_score)

from data import dataset
from evaluation import visualize as mlplots
from utils.utilities import EarlyStopping, balance_input
from utils.logger import setup_logging, log_resources

class Pipeline():

    def __init__(self, model: torch.nn.Module, labels_name: List, seed: int = 42, results_dir = None):
        """ Pipeline for model training and testing.

        Args:
            model (torch.nn.Module): Model
            labels_name (List[str]): Name of the labels
            seed (int, optional): Random seed. Default to 42.
        """

        self.model = model
        # results folder
        date = datetime.today().strftime('%Y_%m_%d')
        hour = datetime.today().strftime('%H_%M_%S')
        if results_dir:
            self.dst_dir =  Path(f"{results_dir}")
        else:
            self.dst_dir = Path(f"experiments/results/{date}/{hour}_{self.model.model_name}")
            if not self.dst_dir.exists():
                self.dst_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initiating model")

        # check if GPU is available
        self.use_cuda = torch.cuda.is_available()
        logger.info(f"GPU available: {self.use_cuda}")
        log_resources(logger)
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        if self.use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.model.to(self.device)

        # fix all random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False

        # model info
        logger.info(f"Model architecture: {self.model.model_name}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):.0f}")

        self.labels_name = labels_name
        self.num_classes = len(self.labels_name)

    def train(self,
              train_ds: dataset.ECGDataset,
              val_ds: dataset.ECGDataset,
              epochs: int,
              batch_size: int,
              lr: float,
              patience: int = 7,
              delta_stop: float = 0,
              val_every: int = 1,
              half_precision: bool = False) -> None:
        """ Train and evaluates a model

        Args:
            train_ds (ECGDataset): Training dataset
            val_ds (ECGDataset): Validation dataset
            epochs (int): Number of epochs
            batch_size (int): Batch size
            lr (float): Maximum learning rate
            patience (int, optional): Number of epochs before early stopping. Defaults to 7.
            delta_stop (float, optional): Minimum delta for early stopping. Defaults to 0.
            val_every (int, optional): Validation step every X epochs. Defaults to 1.
            half_precision(bool, optional): Train with half precision. Defaults to False.
        """

        logger.info("Training start")
        # setup the early stop
        early_stopping = EarlyStopping(patience=patience,
                                       delta=delta_stop,
                                       path=self.dst_dir.joinpath("checkpoint.pt"),
                                       logger=logger)

        # create the dataloaders
        class_weights = train_ds.class_weights
        train_loader = dataset.make_dataloader(ds=train_ds, batch_size=batch_size)
        val_loader = dataset.make_dataloader(ds=val_ds, batch_size=batch_size)

        # training setup
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=epochs)

        if class_weights is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device))

        predictions = []
        lr_evolution = []
        history = {"epoch": [], "train_time": [], "train_loss": [], "val_time": [], "val_loss": []}
        early_break = False
        val_step_counter = 0
        if half_precision:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Training with half-precision.")

        # Train and val loop
        for epoch in range(epochs):
            # train step
            train_loss = 0
            train_start = time.time()
            self.model.train()
            for param_group in optimizer.param_groups:
                lr_evolution.append(param_group['lr'])
            for data, target in train_loader:
                optimizer.zero_grad()
                if half_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data.to(self.device))
                        loss = loss_fn(output, target.to(self.device))
                else:
                    output = self.model(data.to(self.device))
                    loss = loss_fn(output, target.to(self.device))

                train_loss += loss
                # if epoch%5==0 or epoch == 0:
                #     log_resources(logger)

                if half_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

            train_end = time.time()
            train_loss /= len(train_loader.dataset)

            # val step
            val_loss = 0
            val_start = time.time()
            self.model.eval()
            if val_step_counter % val_every == 0 or epoch == epochs - 1 or early_stopping.early_stop:
                val_step_counter = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = self.model(data.to(self.device))

                        if epoch == epochs - 1 or early_stopping.early_stop:
                            predictions.append(torch.nn.functional.sigmoid(output))

                    val_loss += loss_fn(output, target.to(self.device))
                val_loss /= len(val_loader.dataset)
            else:
                val_step_counter += 1
            val_end = time.time()

            history["epoch"].append(1 + epoch)
            history["train_time"].append(train_end - train_start)
            history["train_loss"].append(train_loss.detach().cpu().item())
            history["val_time"].append(val_end - val_start)
            history["val_loss"].append(val_loss.detach().cpu().item())

            logger.info(", ".join([f"{k}: {v[-1]:.6f}" for k, v in history.items()]))

            if early_break:
                break

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                early_break = early_stopping.early_stop

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.model)

        mlplots.plot_loss_evolution(history["train_loss"], history["val_loss"], dst_dir=self.dst_dir, show=False)

        total_time_min = sum(history["train_time"]) / 60
        logger.info(f"Total training time: {total_time_min:.2f} minutes")

        mlplots.plot_lr_evolution(lrs=lr_evolution, dst_dir=self.dst_dir, show=False)

        # metrics
        predictions = torch.cat(predictions).detach().cpu().numpy()

        self.val_thresholds = mlplots.plot_pr_curves(ground_truth=val_ds.labels_bin,
                                                     predictions=predictions,
                                                     labels_name=self.labels_name,
                                                     data_type="val",
                                                     dst_dir=self.dst_dir,
                                                     show=False)
        logger.info(f"Thresholds: {self.val_thresholds}")
        self.val_thresholds = np.array([self.val_thresholds[l] for l in self.labels_name])
        predictions = (predictions > self.val_thresholds).astype(int)

        mlplots.plot_cm_multilabel(ground_truth=val_ds.labels_bin,
                                   predictions=predictions,
                                   labels_name=self.labels_name,
                                   data_type="val",
                                   dst_dir=self.dst_dir,
                                   show=False)

        self.report(val_ds.labels_bin, predictions, self.labels_name)

    def evaluate(self,
              test_ds: dataset.ECGDataset,
              batch_size: int,) -> None:

        """
        Evaluates a trained model on a test dataset.

        Args:
            test_ds (ECGDataset): The test dataset to evaluate the model on.
            batch_size (int): The number of samples per batch to load.
        """
        logger.info("Evaluation starts..")
        test_loader = dataset.make_dataloader(ds=test_ds, batch_size=batch_size)

        test_start = time.time()
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data.to(self.device))
                predictions.append(torch.nn.functional.sigmoid(output))

        test_end = time.time()
         # metrics
        predictions = torch.cat(predictions).detach().cpu().numpy()

        self.test_thresholds = mlplots.plot_pr_curves(ground_truth=test_ds.labels_bin,
                                                     predictions=predictions,
                                                     labels_name=self.labels_name,
                                                     data_type="test",
                                                     dst_dir=self.dst_dir,
                                                     show=False)
        
        logger.info(f"Thresholds: {self.test_thresholds}")
        self.test_thresholds = np.array([self.test_thresholds[l] for l in self.labels_name])
        predictions = (predictions > self.test_thresholds).astype(int)

        mlplots.plot_cm_multilabel(ground_truth=test_ds.labels_bin,
                                   predictions=predictions,
                                   labels_name=self.labels_name,
                                   data_type="test",
                                   dst_dir=self.dst_dir,
                                   show=False)

        self.report(test_ds.labels_bin, predictions, self.labels_name)



    def predict(self, ds: dataset.ECGDataset, batch_size: int = 512, logits: bool = False) -> np.ndarray:
        """
        Predicts the classes from a given ECG dataset.

        Args:
            test_ds (ECGDataset): The test dataset containing ECG signals.
            batch_size (int, optional): The number of samples per batch to load. Defaults to 512.
            logits (bool, optional): If True, returns the raw logits instead of probabilities. Defaults to False.

        Returns:
            np.ndarray: The model's predictions as a numpy array.
        """

        # Create a DataLoader for the test dataset
        pred_loader = dataset.make_dataloader(ds=ds, batch_size=batch_size)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in pred_loader:
                output = self.model(data.to(self.device))
                if logits:
                    predictions.append(output)
                else:
                    predictions.append(torch.nn.functional.sigmoid(output))
        
        predictions = torch.cat(predictions).detach().cpu().numpy()

        return predictions


    def report(self, y_true, y_pred, label_names):

        # micro: averages metrics across all classes, emphasizing overall performance
        # macro: averages metrics independently for each class, giving equal weight to each class
        accuracy = accuracy_score(y_true, y_pred)

        # proportion of predicted positive cases that are actually positive across all classes
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        # proportion of actual positive cases that are correctly predicted as positive across all classes
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # measures the fraction of labels that are incorrectly predicted
        h_loss = hamming_loss(y_true, y_pred)

        # measures similarity between the predicted and true label sets
        jaccard = jaccard_score(y_true, y_pred, average='samples')

        class_report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)

        logger.info("Overall Metrics:")
        logger.info(f"Accuracy: {accuracy:.3f} (higher is better)")
        logger.info(f"Precision (Micro): {precision_micro:.3f} (higher is better)")
        logger.info(f"Precision (Macro): {precision_macro:.3f} (higher is better)")
        logger.info(f"Precision (Weighted): {precision_weighted:.3f} (higher is better)")
        logger.info(f"Recall (Micro): {recall_micro:.3f} (higher is better)")
        logger.info(f"Recall (Macro): {recall_macro:.3f} (higher is better)")
        logger.info(f"Recall (Weighted): {recall_weighted:.3f} (higher is better)")
        logger.info(f"F1-Score (Micro): {f1_micro:.3f} (higher is better)")
        logger.info(f"F1-Score (Macro): {f1_macro:.3f} (higher is better)")
        logger.info(f"F1-Score (Weighted): {f1_weighted:.3f} (higher is better)")
        logger.info(f"Hamming Loss: {h_loss:.3f} (lower is better)")
        logger.info(f"Jaccard Score: {jaccard:.3f} (higher is better)")

        logger.info(f"Classification Report:\n{class_report}")
