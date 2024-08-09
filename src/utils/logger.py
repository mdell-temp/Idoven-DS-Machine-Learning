import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path


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

    