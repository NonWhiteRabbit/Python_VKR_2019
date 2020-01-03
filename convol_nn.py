"""
Пример сверточной нейросети для распознования рукописных цифр с использованием датасета MNIST
"""


import torch
import random
import torchvision.datasets
import numpy as np



# перемешивание данных для воспроизводимости экперимента
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# импорт датасета
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 1-й сверточный слой
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.ELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 2-й сверточный слой
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.ELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 1-й полносвязанный слой
        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = torch.nn.ELU()

        # 2-й полносвязанный слой
        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.ELU()

        # 3-й полносвязанный слой
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """
        Функция прогона тензора через слои
        :param x: тензор (батч из картинок)
        :return: X
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


lenet5 = LeNet5()

# перекладка обработки на GPU, в случае ее наличия, в случае отсутствия остается на CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lenet5 = lenet5.to(device)

# определение фуенции потерь
loss = torch.nn.CrossEntropyLoss()

# определение оптимизатора
optimizer = torch.optim.Adam(lenet5.parameters(), lr=1.0e-4)

# определение размера батча
batch_size = 100

# запись значений точности определения для возможности вывода в виде графика
test_accuracy_history = []

# запись значений потерь для возможности вывода в виде графика
test_loss_history = []

X_test = X_test.to(device)
y_test = y_test.to(device)

# цикл обучения сети
for epoch in range(10000):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds = lenet5.forward(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = lenet5.forward(X_test)
    # test_loss_history.append(loss(test_preds, y_test).data.cpu())

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
    # test_accuracy_history.append(accuracy)

    print(accuracy)