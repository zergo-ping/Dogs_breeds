
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
from ML import *





# Функция для вычисления точности
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels.data).item() / len(labels)

# Обучение с tqdm и ранней остановкой
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_acc = 0.0
epochs_no_improve = 0  # Счетчик эпох без улучшения
early_stop = False  # Флаг для ранней остановки

for epoch in range(EPOCHS):
    if early_stop:
        print(f"\nEarly stopping triggered after epoch {epoch}!")
        break
    
    # Тренировка
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    # Прогресс-бар для тренировки
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', leave=False)
    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Обновляем статистику
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels) * images.size(0)
        
        # Обновляем прогресс-бар
        train_bar.set_postfix({
            'loss': f"{running_loss/(train_bar.n+1):.4f}",
            'acc': f"{running_acc/(train_bar.n+1):.4f}"
        })
    
    train_loss = running_loss / len(train_dataset)
    train_acc = running_acc / len(train_dataset)
    
    # Валидация
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    # Прогресс-бар для валидации
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]', leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            val_acc += accuracy(outputs, labels) * images.size(0)
            
            val_bar.set_postfix({
                'val_loss': f"{val_loss/(val_bar.n+1):.4f}",
                'val_acc': f"{val_acc/(val_bar.n+1):.4f}"
            })
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_acc / len(val_dataset)
    
    # Печатаем статистику по эпохе
    print(f'\nEpoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Логика ранней остановки и сохранения лучшей модели
    if val_acc > best_acc + MIN_DELTA:
        best_acc = val_acc
        torch.save(model.state_dict(), '.\AI_Project\\best_dog_model.pth')
        print('Model saved!')
        epochs_no_improve = 0  # Сбрасываем счетчик
    else:
        epochs_no_improve += 1
        print(f'No improvement for {epochs_no_improve} epochs')
        
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            early_stop = True
    
    scheduler.step()

print(f'\nTraining complete! Best Val Acc: {best_acc:.4f}')