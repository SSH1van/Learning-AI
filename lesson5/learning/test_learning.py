import os
import torch
import torchvision
from torch import nn
from torchvision import transforms


############################################################
#################### Загрузка датасета #####################
############################################################
transform = transforms.Compose(
[transforms.ToTensor(), #Библиотека работает с тензорами поэтому это преобразование в тензор
transforms.Normalize((0.5,), (0.5,))]) #Приводим входные данные к виду [0..1]
# Загружаем БД для обучения и теста
training_set = torchvision.datasets.FashionMNIST('..\\data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('..\\data', train=False, transform=transform, download=True)
# Создаем объекты data loaders для нашего датасета - shuffle=true перемешивает датасет
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)
# Мы будем распознавать классы
classes = ('Футболка', 'Штаны' , 'Пуловер', 'Платье', 'Пальто',
'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки')


############################################################
#################### Создание нейросети ####################
############################################################
import torch.nn as nn
import torch.nn.functional as F
# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        # Тут создаем слои
        self.conv1 = nn.Conv2d(1, 6, 5) #сверточный слой
        self.pool = nn.MaxPool2d(2, 2) #пулинг
        self.conv2 = nn.Conv2d(6, 16, 5) #сверточный слой
        self.fc1 = nn.Linear(16 * 4 * 4, 120) #линейный слой
        self.fc2 = nn.Linear(120, 84) #линейный слой
        self.fc3 = nn.Linear(84, 10) #линейный слой
    def forward(self, x):
        # Тут строим граф
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = GarmentClassifier().cuda()

# Задаём функцию потерь
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MultiMarginLoss()
# loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
# Оптимизатор
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


############################################################
#################### Обучение нейросети ####################
############################################################
# Обучение одной эпохи
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    # Цикл по батчам
    for i, data in enumerate(training_loader):
        # Загружаем данные
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # Обнуляем градиенты в начале итерации
        optimizer.zero_grad()
        # Прямой проход
        outputs = model(inputs)
        # Вычисляем loss
        loss = loss_fn(outputs, labels)
        # Обратный проход
        loss.backward()
        # Меняем веса
        optimizer.step()
        # Считаем средний лосс и выводим его на экран каждую 1000 итераций
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0

    return last_loss

# Обучение нейросети
epoch_number = 0
EPOCHS = 5
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.cuda()
            vlabels = vlabels.cuda()
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    epoch_number += 1


save_path = os.path.join('models', 'Gradient_Adam_model.pth')
torch.save(model.state_dict(), save_path)