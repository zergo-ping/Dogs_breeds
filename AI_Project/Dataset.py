
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ML import *
from ML import BATCH_SIZE
from ML import CSV_PATH
from ML import DATA_DIR

# Dataset класс
class DogDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Аугментация данных
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка и подготовка меток
labels = pd.read_csv(CSV_PATH)
labels['breed'] = pd.Categorical(labels['breed']).codes.astype(np.int64)

# Разделение данных
train_data, val_data = train_test_split(labels, test_size=0.2, random_state=42)

# DataLoader
train_dataset = DogDataset(train_data, DATA_DIR, train_transform)
val_dataset = DogDataset(val_data, DATA_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)