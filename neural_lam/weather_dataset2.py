import datetime as dt
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from . import utils  # Assuming you have some utility functions here

class WeatherDataset(torch.utils.data.Dataset):
    """
    Dataset for handling weather data with 5-minute time steps, 
    but loading thJe peux venir à 15h mardi demain.e data at 1-hour intervals.
    
    The new data format is assumed to be daily files, e.g., 'nwp_20180101.npy', 
    with a typical shape of (288, 565, 784) where 288 corresponds to 5-minute intervals (24 hours).
    """
    
    def __init__(self, dataset_name, pred_length=19, split="train", subsample_step=12, 
                 standardize=True, subset=False, control_only=False):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        # File pattern for daily weather files
        member_file_regexp = "nwp*.npy"
        sample_paths = glob.glob(os.path.join(self.sample_dir_path, member_file_regexp))
        self.sample_names = [path.split("/")[-1][4:-4] for path in sample_paths] # Format: '20180101'

        if subset:
            self.sample_names = self.sample_names[:50]  # Limit to 50 samples if subset is True

        self.subsample_step = subsample_step  # We take every 12th step (1 hour instead of 5 minutes)
        self.standardize = standardize

        # Loading dataset statistics for standardization if required
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = ds_stats["data_mean"], ds_stats["data_std"]

        self.random_subsample = split == "train"

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        sample_path = os.path.join(self.sample_dir_path, f"nwp_{sample_name}.npy")

        # Load the daily weather data
        try:
            full_sample = torch.tensor(np.load(sample_path), dtype=torch.float32)  # (288, 565, 784) or less
        except ValueError:
            print(f"Failed to load {sample_path}")

        # Subsample the data to convert from 5-minute steps to 1-hour steps (every 12th step)
        sample = full_sample[::self.subsample_step]  # (24, 565, 784) for full days

        # If the file contains less than 288 time steps, handle it for partial days
        N_t = sample.shape[0]  # Could be less than 24 if it's the last day of the month

        # Flatten spatial dimensions (565, 784) into a single grid dimension
        sample = sample.flatten(1, 2)  # (N_t, 565*784)

        # Standardize the data if needed
        if self.standardize:
            sample = (sample - self.data_mean) / self.data_std

        # Generate the initial states and target states
        init_states = sample[:2]  # First 2 time steps for initialization
        target_states = sample[2:]  # The remaining time steps are the prediction targets

        return init_states, target_states
