import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
    
import psutil
import GPUtil
import torch

## DATA UTILS ##

from pathlib import Path
from torch.utils.data import DataLoader
from modules.datasets import ECGDataset
from modules.data_preprocessor import ECGDataPreprocessor 

def preprocess_data(raw_data, sample_rate, shuffle= False, normalization_type = 'standardize', segment_length = 0, apply_denoising = False, apply_augmentation=False):
    """Preprocess data for train, val, and test splits."""
    preprocessors = {}
    split_types = ['train', 'val', 'test']
    
    for ds_type in split_types:
        

        if ds_type == 'train':
            preprocessors[ds_type] = ECGDataPreprocessor( #Extended
                annotations_df=raw_data["annotations_df"],
                ecg_signals=raw_data["ecg_signals"],
                ds_type=ds_type,
                shuffle=True,  
                normalization_type=normalization_type,
                segment_length=0,
                sample_rate=sample_rate,
                apply_denoising=apply_denoising,
                apply_augmentation=apply_augmentation
            )
            # Save normalization values for later use
            normalization_value = preprocessors[ds_type].normalization_value
        else:
            preprocessors[ds_type] = ECGDataPreprocessor( # Extended
                annotations_df=raw_data["annotations_df"],
                ecg_signals=raw_data["ecg_signals"],
                ds_type=ds_type,
                shuffle=False,  
                normalization_type=normalization_type,
                normalization_value=normalization_value, 
                segment_length=0,
                sample_rate=sample_rate,
                apply_denoising=False,
                apply_augmentation=False
            )
    data_info = {
                'labels_bin': preprocessors['train'].labels_bin,
                'labels_class': preprocessors['train'].labels_class,
                'signal_clean': preprocessors['train'].signal_clean,
                'num_classes': preprocessors['train'].num_classes,
                'channels': preprocessors['train'].channels
            }

    return preprocessors, data_info

def create_dataloaders(preprocessors, batch_size):
    """Create DataLoaders for train, val, and test splits."""
    dataloaders = {}
    
    for ds_type, preprocessor in preprocessors.items():
        dataloaders[ds_type] = DataLoader(
            preprocessor,
            batch_size=batch_size,
            shuffle=preprocessor.shuffle,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
    
    return dataloaders

## Experiments UTILS ##

def setup_logging(log_file: str = "logger.log", experiment_dir: Path = None):

    if experiment_dir:
        dst_dir = experiment_dir
        dst_file = dst_dir.joinpath(log_file)
    else:
        date = datetime.today().strftime('%Y_%m_%d')
        hour = datetime.today().strftime('%H_%M_%S')
        dst_dir = Path(f"experiments/logs/{date}/{hour}")
        dst_file = dst_dir.joinpath(log_file)
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(dst_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_resources(logger):
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    gpu_info = GPUtil.getGPUs()
    
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(f"Memory Usage: {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
    
    if gpu_info:
        for gpu in gpu_info:
            logger.info(f"GPU {gpu.id} - Usage: {gpu.load * 100:.2f}%, Memory Usage: {gpu.memoryUtil * 100:.2f}% (Free: {gpu.memoryFree / (1024**3):.2f} GB)")
    else:
        logger.info("No GPU detected")


def print_gpu_memory_usage():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU and CUDA installation.")
        return

    device = torch.device('cuda')
    print(f"\tMemory usage for GPU: {torch.cuda.get_device_name(device)}")

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    print(f"\t\tAllocated memory: {allocated / (1024 ** 2):.2f} MB")
    print(f"\t\tReserved memory: {reserved / (1024 ** 2):.2f} MB")
    print(f"\t\tMax allocated memory: {max_allocated / (1024 ** 2):.2f} MB")
    print(f"\t\tMax reserved memory: {max_reserved / (1024 ** 2):.2f} MB")

def check_torch_cuda_installation(print_memory_usage = False):

    print('Check torch-cuda installation: ')
    cuda_available = torch.cuda.is_available()
    print('Cuda available: ', cuda_available)

    if cuda_available:

        print('\t# CUDA devices: ', torch.cuda.device_count())

        print('\tCurrent device: ',torch.cuda.current_device())
        print(f'\tDevice information:\n\t\t{torch.cuda.device(0)}\n\t\t{torch.cuda.get_device_name(0)}')
    if print_memory_usage:
        print_gpu_memory_usage()