import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# The 14 official NIH findings
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class NIHDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None): 
        """
        Args:
            csv_path (str): Path to the train/val/test CSV file.
            img_dir (str): Path to the folder containing all images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Extract the 14 labels as a vector (e.g., [0, 0, 1, ...])
        label_vec = row[self.labels].values.astype(np.float32)

        try:
            # Ensure 3-channel RGB (even if original is grayscale)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback: load the next image to avoid crashing training
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vec)