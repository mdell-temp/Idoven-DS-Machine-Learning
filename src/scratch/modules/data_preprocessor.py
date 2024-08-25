from typing import List, Union
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
import pywt
import random

class ECGDataPreprocessor(Dataset): # Extended
    def __init__(self, annotations_df: pd.DataFrame, ecg_signals: np.ndarray, ds_type: str = "train", shuffle: bool = False, 
                 normalization_type: str = "standardize", normalization_value: dict = None, segment_length: int = 0, sample_rate: int = 100, 
                 apply_denoising: bool = True, apply_augmentation: bool = False):
        """
        Initialize the ECG Data Preprocessor with additional preprocessing steps.

        Args:
            annotations_df (pd.DataFrame): DataFrame containing annotation data.
            ecg_signals (np.ndarray): Loaded ECG signals.
            ds_type (str): Specify the type of dataset - "train", "val", or "test".
            shuffle (bool): Whether to shuffle the data (only for training).
            normalization_type (str): Type of normalization to apply - "standardize" or "minmax".
            normalization_value (dict): Dictionary containing normalization parameters (mean, std for standardize or min-max values).
            segment_length (int): Length of each ECG segment (in samples).
            sample_rate (int): Sampling rate for the ECG signals.
            apply_denoising (bool): Whether to apply denoising to the ECG signals.
            apply_augmentation (bool): Whether to apply data augmentation techniques.
        """
        self.annotations_df = annotations_df
        self.ecg_signals = ecg_signals
        self.ds_type = ds_type
        self.shuffle = shuffle
        self.normalization_type = normalization_type
        self.normalization_value = normalization_value
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.apply_denoising = apply_denoising
        self.apply_augmentation = apply_augmentation
        
        # Data containers
        self.labels_bin = None
        self.labels_class = None
        self.signal_clean = None
        self.num_classes = None
        self.channels = None

        # Preprocess data
        self._split_data()
        self._prepare_dataset()
        
        # Apply additional preprocessing steps
        if self.apply_denoising:
            self._apply_denoising()
        if self.apply_augmentation:
            self._augment_data()

    def _split_data(self) -> None:
        """Split data into training, validation, and test sets."""
        subsets = {
            'train': self.annotations_df.strat_fold <= 8,
            'val': self.annotations_df.strat_fold == 9,
            'test': self.annotations_df.strat_fold == 10
        }

        self.signals = self.ecg_signals[subsets[self.ds_type]]
        self.labels = self.annotations_df[subsets[self.ds_type]]['diagnostic_superclass']

    def _prepare_dataset(self) -> None:
        """Prepare the dataset for the specified data split."""
        # Remove data points without labels
        self.signals, self.labels = self._remove_no_label_data(self.signals, self.labels)

        # Binarize the labels
        self.labels_bin, self.labels_class = self._binarize_labels(self.labels)

        # Normalize the signals
        self.signal_clean = self._normalize_signal_per_channel(self.signals)

        # Standardize signal length
        if self.segment_length != 0:
            self._standardize_signal_length()

        # Set additional attributes
        self.num_classes = self.labels_bin.shape[1]
        self.channels = self.signal_clean.shape[2]

        # Shuffle the dataset if requested (only for training)
        if self.shuffle:
            permutation = np.random.permutation(self.signal_clean.shape[0])
            self.signal_clean = self.signal_clean[permutation, :, :]
            self.labels_bin = self.labels_bin[permutation, :]

    def _normalize_signal_per_channel(self, signal: np.ndarray) -> np.ndarray:
        """Normalize the signal per channel based on the selected normalization type."""
        if self.normalization_type == "standardize":
            if self.normalization_value is None:
                return self._standardize_signal_per_channel(signal)
            else:
                mean, std = self.normalization_value['mean'], self.normalization_value['std']
                return self._apply_standardization_with_values(signal, mean, std)
        elif self.normalization_type == "minmax":
            if self.normalization_value is None:
                return self._minmax_normalize_signal_per_channel(signal)
            else:
                min_val, max_val = self.normalization_value['min'], self.normalization_value['max']
                return self._apply_minmax_normalization_with_values(signal, min_val, max_val)
        else:
            raise ValueError("Unsupported normalization type. Choose 'standardize' or 'minmax'.")

    def _apply_standardization_with_values(self, signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply standardization using precomputed mean and std."""
        channels = signal.shape[-1]
        for i in range(channels):
            signal[:, :, i] = (signal[:, :, i] - mean[i]) / std[i]
        return signal

    def _apply_minmax_normalization_with_values(self, signal: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Apply min-max normalization using precomputed min and max values."""
        channels = signal.shape[-1]
        for i in range(channels):
            signal[:, :, i] = (signal[:, :, i] - min_val[i]) / (max_val[i] - min_val[i])
        return signal

    def _standardize_signal_per_channel(self, signal: np.ndarray) -> np.ndarray:
        """Standardize the signal per channel and compute normalization parameters."""
        channels = signal.shape[-1]
        mean = np.array([np.mean(signal[:, :, i], keepdims=True) for i in range(channels)])
        std = np.array([np.std(signal[:, :, i], keepdims=True) for i in range(channels)])
        for i in range(signal.shape[-1]):
            signal[:, :, i] = (signal[:, :, i] - mean[i]) / std[i]
        self.normalization_value = {'mean': mean.flatten(), 'std': std.flatten()}
        # Check for NaNs or Infs
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            raise ValueError("Standardized signal contains NaNs or Infs")
        return signal

    def _minmax_normalize_signal_per_channel(self, signal: np.ndarray) -> np.ndarray:
        """Apply min-max normalization per channel and compute normalization parameters."""
        channels = signal.shape[-1]
        min_val = np.array([np.min(signal[:, :, i]) for i in range(channels)])
        max_val = np.array([np.max(signal[:, :, i]) for i in range(channels)])
        for i in range(channels):
            signal[:, :, i] = (signal[:, :, i] - min_val[i]) / (max_val[i] - min_val[i])
        self.normalization_value = {'min': min_val, 'max': max_val}
        return signal

    def _standardize_signal_length(self) -> None:
        """Standardize ECG signal lengths with zero padding or truncation."""
        standardized_signals = []
        for signal in self.signal_clean:
            if len(signal) < self.segment_length:
                # Zero padding
                padded_signal = np.pad(signal, ((0, self.segment_length - len(signal)), (0, 0)), 'constant')
                standardized_signals.append(padded_signal)
            else:
                # Truncate to the segment length
                standardized_signals.append(signal[:self.segment_length])
        self.signal_clean = np.array(standardized_signals)

    def _apply_denoising(self) -> None:
        """Apply bandpass filtering and wavelet denoising to the ECG signals."""
        denoised_signals = []
        for signal in self.signal_clean:
            # Apply bandpass filtering
            bandpassed_signal = self._bandpass_filter(signal, self.sample_rate)

            # Apply wavelet denoising
            denoised_signal = self._wavelet_denoising(bandpassed_signal)
            denoised_signals.append(denoised_signal)
        self.signal_clean = np.array(denoised_signals)

    def _bandpass_filter(self, signal: np.ndarray, sample_rate: int = 100, lowcut: float = 0.5, highcut: float = 40.0, order: int = 1) -> np.ndarray:
        """Apply a bandpass filter to the signal."""
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=0)

    def _wavelet_denoising(self, signal: np.ndarray, wavelet: str = 'db4', level: int = 1) -> np.ndarray:
        """Apply wavelet denoising to the signal."""
        coeffs = pywt.wavedec(signal, wavelet, mode='per')
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet, mode='per')

    def _augment_data(self) -> None:
        """Apply data augmentation techniques such as time warping, amplitude scaling, and adding noise."""
        augmented_signals = []
        augmented_labels = []
        for signal, label in zip(self.signal_clean, self.labels_bin):
            # Original signal
            augmented_signals.append(signal)
            augmented_labels.append(label)
            
            # Apply time warping
            warped_signal = self._time_warp(signal)
            augmented_signals.append(warped_signal)
            augmented_labels.append(label)

            # Apply amplitude scaling
            scaled_signal = self._amplitude_scaling(signal)
            augmented_signals.append(scaled_signal)
            augmented_labels.append(label)

            # Apply noise
            noisy_signal = self._add_noise(signal)
            augmented_signals.append(noisy_signal)
            augmented_labels.append(label)
        
        self.signal_clean = np.array(augmented_signals)
        self.labels_bin = np.array(augmented_labels)
        
    @staticmethod
    def _time_warp(signal: np.ndarray, max_warp: float = 0.2) -> np.ndarray:
        """Apply time warping to the signal."""
        if len(signal.shape) == 2:  # (samples, channels)
            num_samples, num_channels = signal.shape
            warped_signal = np.zeros_like(signal)
            for i in range(num_channels):
                time_warp_factor = 1 + (random.random() - 0.5) * 2 * max_warp
                # Apply interpolation per channel
                warped_signal[:, i] = np.interp(
                    np.arange(num_samples) * time_warp_factor,
                    np.arange(num_samples),
                    signal[:, i],
                    left=0,
                    right=0
                )
            return warped_signal
        else:
            raise ValueError("Signal shape not supported for time warping.")


    @staticmethod
    def _amplitude_scaling(signal: np.ndarray, scaling_factor: float = 0.1) -> np.ndarray:
        """Apply amplitude scaling to the signal."""
        scaling = 1 + (random.random() - 0.5) * 2 * scaling_factor
        return signal * scaling

    @staticmethod
    def _add_noise(signal: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to the signal."""
        noise = np.random.normal(0, noise_factor, signal.shape)
        return signal + noise

    @staticmethod
    def _remove_no_label_data(data, labels):
        """Remove data points without labels."""
        valid_idx = [i for i, label in enumerate(labels) if isinstance(label, list) and len(label) > 0]
        return data[valid_idx], labels.iloc[valid_idx]

    @staticmethod
    def _binarize_labels(multi_labels: pd.Series) -> Union[np.ndarray, List]:
        """Binarize labels."""
        mlb = MultiLabelBinarizer()
        mlb.fit(multi_labels)
        bin_labels = mlb.transform(multi_labels)
        return bin_labels, mlb.classes_

    def __len__(self):
        return self.signal_clean.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.signal_clean[idx]).float()
        label = torch.from_numpy(self.labels_bin[idx]).float()
        return sample, label
