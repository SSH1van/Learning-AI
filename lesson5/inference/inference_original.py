import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image

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

# Трансформации
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Создаём экземпляр модели
model = GarmentClassifier()

# Загружаем сохранённые параметры
load_path = os.path.join('models', 'base_model.pth')
model.load_state_dict(torch.load(load_path))

# Переключаем модель в режим оценки
model.eval()

# Путь к папке с изображениями
image_folder = './original_images'

# Проходим по всем изображениям в папке
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    # Проверяем, что это файл изображения
    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Загружаем изображение
            image = Image.open(image_path)
            
            # Применяем трансформации
            image_tensor = transform(image)
            
            # Визуализация преобразованного изображения
            # Денормализуем изображение для отображения (обратное преобразование нормализации)
            display_image = image_tensor.squeeze(0)  # Убираем размер батча
            display_image = display_image * 0.5 + 0.5  # Денормализация: (x * std + mean)
            display_image = display_image.numpy()  # Преобразуем в numpy для matplotlib
            
            # Добавляем размер батча для предсказания
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