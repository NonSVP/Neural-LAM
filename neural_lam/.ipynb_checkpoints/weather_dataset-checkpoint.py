import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = self.get_valid_files()
        
    def get_valid_files(self):
        # List all files in directory
        files = os.listdir(self.data_dir)
        valid_files = []
        expected_shape = (288, 565, 784)  # Expected shape for valid files
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            try:
                # Check if file shape matches expected shape
                data = np.load(file_path)
                if data.shape == expected_shape:
                    valid_files.append(file_path)
                else:
                    print(f"File {file} skipped due to shape mismatch. Found: {data.shape}, Expected: {expected_shape}")
            except Exception as e:
                print(f"Could not load file {file}. Error: {e}")
        
        return valid_files
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        return data

# Custom collate function to handle batch data
def custom_collate_fn(batch):
    # Implement custom handling here if necessary
    return torch.tensor(batch)

def main():
    data_dir = './data'  # Replace with your dataset directory path
    dataset = CustomDataset(data_dir)
    
    # Using smaller batch size and fewer workers to avoid memory overload
    loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)

    print("Saving parameter weights...")
    print("Computing mean and std.-dev. for parameters...")

    # Loop through data loader
    for init_batch in tqdm(loader):
        # Your processing code here
        # Example: 
        # mean = torch.mean(init_batch, dim=0)
        # std = torch.std(init_batch, dim=0)
        pass

if __name__ == '__main__':
    main()
