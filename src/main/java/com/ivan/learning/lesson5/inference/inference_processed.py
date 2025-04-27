import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import numpy as np

# Определяем класс модели
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Названия классов
classes = ('Футболка', 'Штаны', 'Пуловер', 'Платье', 'Пальто',
           'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки')

# Создаём экземпляр модели
model = GarmentClassifier()

# Загружаем сохранённые параметры
load_path = os.path.join('models', 'base_model.pth')
model.load_state_dict(torch.load(load_path))

# Переключаем модель в режим оценки
model.eval()

# Путь к папке с изображениями
image_folder = './processed_images'

# Проходим по всем изображениям в папке
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    # Проверяем, что это файл изображения
    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Загружаем изображение
            image = Image.open(image_path)
            
            # 1. Преобразование в оттенки серого
            image = image.convert('L')  # 'L' — режим оттенков серого (1 канал)
            
            # 2. Преобразование в тензор
            # Сначала преобразуем изображение в numpy массив
            image_np = np.array(image, dtype=np.float32)
            # Преобразуем в тензор и добавляем размер канала (1, H, W)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Форма: (1, 28, 28)
            
            # 3. Нормализация
            image_tensor = image_tensor / 255.0  # Масштабируем пиксели в [0, 1]
            image_tensor = (image_tensor - 0.5) / 0.5  # Нормализация: (x - mean) / std
            
            # Визуализация преобразованного изображения
            # Денормализуем для отображения
            display_image = image_tensor.squeeze(0)  # Убираем размер канала
            display_image = display_image * 0.5 + 0.5  # Денормализация
            display_image = display_image.numpy()  # Преобразуем в numpy для matplotlib
            
            # Добавляем размер батча для предсказания (1, 1, 28, 28)
            image_tensor = image_tensor.unsqueeze(0)
            
            # Делаем предсказание
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
            
            # Отображаем изображение с предсказанием
            plt.figure(figsize=(5, 5))
            plt.imshow(display_image, cmap='gray')
            plt.title(f'{image_name}\nПредсказанный класс: {classes[predicted.item()]}', fontsize=14)
            plt.axis('off')
            plt.show()
        
        except Exception as e:
            print(f'Ошибка при обработке {image_name}: {e}')