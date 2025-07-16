from PIL import Image  
from matplotlib import pyplot as plt
import torch
from torchvision import transforms, models
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загрузка данных
labels_df = pd.read_csv('.\AI_Project\dog_breeds\labels.csv')  # Колонки: 'image_id', 'breed'

path_of_file = '.\AI_Project\dog_breeds\\test\\f5174e3ab52edb2e414e135259047c48.jpg'

# Создаем кодировщик и словарь
le = LabelEncoder()
labels_df['breed_encoded'] = le.fit_transform(labels_df['breed'])
idx_to_breed = {idx: breed for idx, breed in enumerate(le.classes_)}
NUM_CLASSES = len(idx_to_breed)  

# Сохраняем словарь
import pickle
with open('idx_to_breed.pkl', 'wb') as f:
    pickle.dump(idx_to_breed, f)

# Инициализация модели с правильным числом классов
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES) 
model = model.to(DEVICE)

# Загрузка весов
try:
    checkpoint = torch.load('best_dog_model.pth')
    
    # Проверка совместимости классов
    if checkpoint.get('num_classes') != NUM_CLASSES:
        print(f"Внимание: модель обучена на {checkpoint.get('num_classes')} классов, а текущих {NUM_CLASSES}")
    
    # Загрузка с обработкой несоответствий
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
except:
    pretrained_dict = torch.load('.\AI_Project\\best_dog_model.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

model.eval()

# Трансформер
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Функция предсказания
def test_single_image(model, image_path, transform, idx_to_breed, device='cuda'):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            pred_idx = predicted.item()
            
            # Защита от отсутствующего ключа
            breed_name = idx_to_breed.get(pred_idx, f"unknown_breed_{pred_idx}")
            confidence = prob[pred_idx].item()
        
        plt.imshow(image)
        plt.title(f"Predicted: {breed_name}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")

# Пример использования
test_single_image(
    model,
    path_of_file,
    val_transform,
    idx_to_breed,
    DEVICE
)