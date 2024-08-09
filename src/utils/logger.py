import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
    
import psutil
import GPUtil
import torch

def setup_logging(log_file: str = "logger.log"):

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
    