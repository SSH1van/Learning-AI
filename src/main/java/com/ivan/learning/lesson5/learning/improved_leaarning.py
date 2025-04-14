import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

# Трансформации без изменений
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка датасета
training_set = torchvision.datasets.FashionMNIST('..\\data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('..\\data', train=False, transform=transform, download=True)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)  # Изменён batch_size
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

classes = ('Футболка', 'Штаны', 'Пуловер', 'Платье', 'Пальто',
           'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки')

# Модель с добавлением Dropout
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)          # Ядро 3x3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)         # Ядро 3x3
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # Исправлено на 5x5x16
        self.dropout1 = nn.Dropout(0.5)          # Добавлен Dropout
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)          # Добавлен Dropout
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)              # Исправлено на 5x5x16
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = GarmentClassifier().cuda()

# Функция потерь
loss_fn = nn.CrossEntropyLoss()

# Оптимизатор Adam с weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Обучение одной эпохи
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
    return last_loss

# Обучение нейросети
epoch_number = 0
EPOCHS = 10  # Увеличено количество эпох
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

# Сохранение модели
save_path = os.path.join('models', 'improved_model.pth')
torch.save(model.state_dict(), save_path)