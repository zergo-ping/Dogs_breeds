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

from Dataset import *

# Конфигурация
DATA_DIR = '.\AI_Project\dog_breeds\\train'
CSV_PATH = '.\AI_Project\dog_breeds\labels.csv'
NUM_CLASSES = 120
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры для ранней остановки
EARLY_STOP_PATIENCE = 3  # Количество эпох без улучшения перед остановкой
MIN_DELTA = 0.001  # Минимальное значимое улучшение точности

# Модель с BatchNorm и Dropout
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)
if __name__ == "__main__":
    print("setset")

