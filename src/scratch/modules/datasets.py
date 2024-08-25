# ecg_dataset_loader.py
import logging
from pathlib import Path
from typing import Dict

import ast
import pandas as pd
import wfdb
import numpy as np

logger = logging.getLogger(__name__)

class ECGDataset:
    def __init__(self, data_dir: Path = Path('physionet.org/files/ptb-xl/1.0.2'), sample_rate: int = 100):
        """
        Initialize the ECG Dataset Loader.

        Args:
            data_dir (Path): Path to the ECG data directory.
            sample_rate (int): Sample rate for the ECG signals.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.annotations_df = None
        self.ecg_signals = None
        self.diagnostic_info = None

        # Load data
        self.load_data()

    def load_data(self) -> None:
        """Load the ECG data and associated annotations."""
        self._verify_files()
        self._load_annotations()
        self._load_diagnostics()
        self._map_diagnostic_superclasses()
        self._load_ecg_signals()

    def _verify_files(self) -> None:
        """Ensure all required files exist in the dataset directory."""
        annotation_file = self.data_dir / "ptbxl_database.csv"
        diagnostic_file = self.data_dir / "scp_statements.csv"

        if not annotation_file.exists() or not diagnostic_file.exists():
            raise FileNotFoundError("Required files are missing in the specified directory.")

    def _load_annotations(self) -> None:
        """Load the annotation data."""
        annotation_file = self.data_dir / "ptbxl_database.csv"
        self.annotations_df = pd.read_csv(annotation_file, index_col='ecg_id')
        self.annotations_df['scp_codes'] = self.annotations_df['scp_codes'].apply(ast.literal_eval)
        self.annotations_df.sex = self.annotations_df.sex.map({0: 'male', 1: 'female'})

    def _load_diagnostics(self) -> None:
        """Load the diagnostic classifications."""
        diagnostic_file = self.data_dir / "scp_statements.csv"
        self.diagnostic_info = pd.read_csv(diagnostic_file, index_col=0)
        self.diagnostic_info = self.diagnostic_info[self.diagnostic_info.diagnostic == 1]

    def _map_diagnostic_superclasses(self) -> None:
        """Map SCP codes to diagnostic superclasses."""
        diagnostic_superclass = []
        for codes in self.annotations_df['scp_codes']:
            classes = set()
            for code in codes:
                if code in self.diagnostic_info.index:
                    classes.add(self.diagnostic_info.at[code, 'diagnostic_class'])
            diagnostic_superclass.append(list(classes))
        self.annotations_df['diagnostic_superclass'] = diagnostic_superclass

    def _load_ecg_signals(self) -> None:
        """Load ECG signal data."""
        file_key = 'filename_lr' if self.sample_rate == 100 else 'filename_hr'
        signal_files = self.annotations_df[file_key].apply(lambda fname: wfdb.rdsamp(self.data_dir / fname)[0])
        self.ecg_signals = np.stack(signal_files.values)

    def get_raw_data(self) -> Dict[str, any]:
        """Retrieve raw loaded data and metadata."""
        return {
            "annotations_df": self.annotations_df,
            "ecg_signals": self.ecg_signals
        }
