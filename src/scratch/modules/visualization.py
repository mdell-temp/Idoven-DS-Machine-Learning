import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import label_binarize
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from typing import List, Dict

logger = logging.getLogger(__name__)

def plot_metrics(history, folder_path, data_type):
    epochs = history['epoch']
    
    # Plot for Losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path.joinpath(f'{data_type}_loss_over_epochs.png'))
    plt.show()
    logger.info("Loss plot saved to loss_over_epochs.png")

    # Plot for Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path.joinpath(f'{data_type}_learning_rate_over_epochs.png'))
    plt.show()
    logger.info("Learning rate plot saved to learning_rate_over_epochs.png")

    # Combined Plot for Accuracy, Precision, Recall, F1-Score
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.plot(epochs, history['val_precision'], label='Validation Precision')
    plt.plot(epochs, history['val_recall'], label='Validation Recall')
    plt.plot(epochs, history['val_f1'], label='Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Validation Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path.joinpath(f'{data_type}_metrics_over_epochs.png'))
    plt.show()
    logger.info("Metrics plots (Accuracy, Precision, Recall, F1-Score) saved to metrics_over_epochs.png")


def plot_and_save_individual_confusion_matrices(true_labels: np.ndarray,
                                                predicted_labels: np.ndarray,
                                                class_names: list,
                                                data_type: str,
                                                save_dir: Path = None):
    """
    Generates and saves individual confusion matrices for each class.

    Args:
        true_labels (np.ndarray): Ground truth binary labels.
        predicted_labels (np.ndarray): Predicted binary labels.
        class_names (list): List of class names corresponding to the labels.
        save_dir (Path): Path to directory where the plots will be saved.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute confusion matrices for each label
    confusion_mats = multilabel_confusion_matrix(true_labels, predicted_labels)

    for i, (cm, class_name) in enumerate(zip(confusion_mats, class_names)):
        cm_frame = pd.DataFrame(cm,
                                index=[f"Actual {class_name}", f"Actual Not {class_name}"],
                                columns=[f"Predicted {class_name}", f"Predicted Not {class_name}"])

        cm_perc = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        # Annots with counts and percentages
        labels = np.array(
            [[f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)

        for x in range(cm.shape[0]):
            for y in range(cm.shape[1]):
                ax.text(y, x, labels[x, y],
                        ha="center", va="center",
                        color="white" if cm[x, y] > cm.max() / 2 else "black")

        ax.set_title(f"Confusion Matrix: {class_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"{class_name}", f"Not {class_name}"])
        ax.set_yticklabels([f"{class_name}", f"Not {class_name}"])

        save_path = save_dir.joinpath(f"{data_type}_confusion_matrix_{class_name}.png")       
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        logger.info(f"Confusion matrix for class '{class_name}' saved at: {save_path}")

def plot_roc_curve(all_targets, all_predictions, results_dir, data_type):
    fpr, tpr, _ = roc_curve(all_targets.numpy().ravel(), all_predictions.numpy().ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(results_dir.joinpath(f'{data_type}_roc_curve.png'))
    plt.show()

    plt.close()
    logger.info("ROC curve saved to roc_curve.png")

def plot_precision_recall_curve(all_targets, all_predictions, results_dir, data_type):
    precision, recall, _ = precision_recall_curve(all_targets.numpy().ravel(), all_predictions.numpy().ravel())
    pr_auc = average_precision_score(all_targets.numpy().ravel(), all_predictions.numpy().ravel())
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.grid(True)        
    plt.savefig(results_dir.joinpath(f'{data_type}_precision_recall_curve.png'))
    plt.show()
    plt.close()
    logger.info("Precision-Recall curve saved to precision_recall_curve.png")

def plot_roc_curve_per_class(all_targets, all_predictions, labels_class, results_dir, data_type):
    num_classes = len(labels_class)
    all_targets_one_hot = label_binarize(all_targets.numpy(), classes=range(num_classes))
    all_predictions = torch.sigmoid(all_predictions).numpy()
    
    plt.figure(figsize=(12, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_targets_one_hot[:, i], all_predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {labels_class[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc='lower right')
    plt.grid(True)        
    plt.savefig(results_dir.joinpath(f'{data_type}_roc_curve_per_class.png'))
    plt.show()
    plt.close()
    logger.info("ROC curve for each class saved to roc_curve_per_class.png")

def plot_precision_recall_curve_per_class(all_targets, all_predictions, labels_class, results_dir, data_type):
    num_classes = len(labels_class)
    all_targets_one_hot = label_binarize(all_targets.numpy(), classes=range(num_classes))
    all_predictions = torch.sigmoid(all_predictions).numpy()
    
    plt.figure(figsize=(12, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_targets_one_hot[:, i], all_predictions[:, i])
        pr_auc = average_precision_score(all_targets_one_hot[:, i], all_predictions[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {labels_class[i]} (AUC = {pr_auc:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc='lower left')
    plt.grid(True)        
    plt.savefig(results_dir.joinpath(f'{data_type}_precision_recall_curve_per_class.png'))
    plt.show()
    plt.close()
    logger.info("Precision-Recall curve for each class saved to precision_recall_curve_per_class.png")


# Visualizzare data

CHANNELS_NAME = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_raw_vs_processed_signals(raw_signal: np.ndarray, processed_signal: np.ndarray, annotation_row: pd.Series):
    """Plot raw and processed ECG signals for comparison."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(raw_signal, label='Raw Signal')
    plt.title('Raw ECG Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.plot(processed_signal, label='Processed Signal')
    plt.title('Processed ECG Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    # Adding annotation to the title
    title_text = (
        f"Diagnostic Superclass: {annotation_row['diagnostic_superclass']}\n"
        f"Age: {annotation_row['age']}\n"
        f"Sex: {annotation_row['sex']}"
    )
    plt.suptitle(title_text, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()



def plot_raw_vs_processed_signals_grid(raw_signal: np.ndarray, processed_signal: np.ndarray, annotation_row: pd.Series):
    """
    Plot raw vs. processed signals in a grid with 2 columns and 6 rows.
    """
    num_channels = raw_signal.shape[1]
    fig, axes = plt.subplots(6, 2, figsize=(15, 18))
    
    for i in range(num_channels):
        row = i // 2
        col = i % 2
        
        axes[row, col].plot(raw_signal[:, i], label='Raw Signal') # raw
        axes[row, col].plot(processed_signal[:, i], label='Processed Signal', linestyle='--') # processed
        
        # Set title and labels using channel names
        axes[row, col].set_title(f'Channel {CHANNELS_NAME[i]}')
        axes[row, col].set_xlabel('Time (samples)')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].legend()
    
    # Adding annotation to the figure title
    title_text = (
        f"Diagnostic Superclass: {annotation_row['diagnostic_superclass']}\n"
        f"Age: {annotation_row['age']}\n"
        f"Sex: {annotation_row['sex']}"
    )
    fig.suptitle(title_text, fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.show()



def plot_amplitude_distribution(signals: np.ndarray, title: str):
    """Plot the distribution of signal amplitudes."""
    sns.histplot(signals.flatten(), bins=100, kde=True)
    plt.title(f'Amplitude Distribution - {title}')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.show()

def plot_psd(signal: np.ndarray, sample_rate: int):
    """Plot the power spectral density (PSD) of the signal."""
    f, Pxx_den = welch(signal, sample_rate, nperseg=1024)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx_den)
    plt.title('Power Spectral Density of ECG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

def plot_psd_multichannel(signal: np.ndarray, sample_rate: int, annotation_row: pd.Series):
    """Plot Power Spectral Density (PSD) for each channel in a multi-channel signal."""
    num_channels = signal.shape[1]
    num_rows = 6
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 18), sharex=True, sharey=True)
    axes = axes.flatten() 
    
    for i in range(num_channels):
        f, Pxx_den = welch(signal[:, i], sample_rate, nperseg=1024)
        axes[i].semilogy(f, Pxx_den)
        axes[i].set_title(f'Channel {CHANNELS_NAME[i]}')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Power/Frequency (dB/Hz)')
    
    # Adding annotation to the figure title
    title_text = (
        f"Diagnostic Superclass: {annotation_row['diagnostic_superclass']}\n"
        f"Age: {annotation_row['age']}\n"
        f"Sex: {annotation_row['sex']}"
    )
    fig.suptitle(title_text, fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.show()

def plot_psd_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, sample_rate: int, annotation_row: pd.Series):
    """Plot Power Spectral Density (PSD) for both raw and processed signals for each channel with annotations."""
    
    num_channels = raw_signal.shape[1]
    num_rows = 6
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 18), sharex=True, sharey=True)
    axes = axes.flatten()  

    for i in range(num_channels):
        f_raw, Pxx_den_raw = welch(raw_signal[:, i], sample_rate, nperseg=1024)
        f_proc, Pxx_den_proc = welch(processed_signal[:, i], sample_rate, nperseg=1024)
        
        axes[i].semilogy(f_raw, Pxx_den_raw, label='Raw Signal', color='blue')
        axes[i].semilogy(f_proc, Pxx_den_proc, label='Processed Signal', color='red', linestyle='--')
        
        axes[i].set_title(f'Channel {CHANNELS_NAME[i]}')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Power/Frequency (dB/Hz)')
        axes[i].legend()
    
    title_text = (
        f"Diagnostic Superclass: {annotation_row['diagnostic_superclass']}\n"
        f"Age: {annotation_row['age']}\n"
        f"Sex: {annotation_row['sex']}"
    )
    fig.suptitle(title_text, fontsize=14, y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()


def plot_class_distribution(labels: np.ndarray, classes: List[str]):
    """Plot the class distribution."""
    sns.countplot(x=labels, order=range(len(classes)))
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_combined_class_distribution_relative(preprocessors: Dict[str, 'Preprocessor'], classes: List[str]):
    """
    Plot the class distribution for multiple datasets (train, val, test) on a single plot.
    Bars are grouped side-by-side for each class and show relative frequencies.
    
    Args:
        preprocessors (Dict[str, Preprocessor]): Dictionary of preprocessors for different datasets.
        classes (List[str]): List of class labels.
    """
    num_classes = len(classes)
    num_datasets = len(preprocessors)
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(num_classes)
    
    for i, (dataset_name, preprocessor) in enumerate(preprocessors.items()):
        
        labels = np.argmax(preprocessor.labels_bin, axis=1)
        counts = np.bincount(labels, minlength=num_classes)
        frequencies = counts / counts.sum()
        
        ax.bar(index + i * bar_width, frequencies, bar_width, label=f'{dataset_name.capitalize()} Data', color=colors[i])
    
    # Set labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Relative Frequency')
    ax.set_title('Class Distribution Comparison')
    ax.set_xticks(index + (num_datasets - 1) * bar_width / 2)
    ax.set_xticklabels(classes)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_signal_with_annotations(ecg_signal: np.ndarray, annotation_row: pd.Series, ):
    num_channels = ecg_signal.shape[1]
    num_rows = 6
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(num_channels):
        axes[i].plot(ecg_signal[:, i], label='ECG Signal', color='blue')
        axes[i].set_title(f'Channel {CHANNELS_NAME[i]}')
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
    
    title_text = (
        f"Diagnostic Superclass: {annotation_row['diagnostic_superclass']}\n"
        f"Age: {annotation_row['age']}\n"
        f"Sex: {annotation_row['sex']}"
    )
    fig.suptitle(title_text, fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.show()
