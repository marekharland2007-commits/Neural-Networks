import torch
import torch.nn as nn

# A simple perceptron neural network for MNIST classification
class MNISTNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

        # He initialization (similar to your NumPy init)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # no softmax (CrossEntropyLoss will handle)
        return x


# CNN architecture neural network for MNIST classification
class CNNNet(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1   = nn.Linear(128 * 3 * 3, 128)
        self.fc2   = nn.Linear(128, 10)


    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.dropout_conv(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x
    

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out

# CNN architecture neural network for EMNIST classification
class UpgradedCNNNet(nn.Module):
    def __init__(self):
        layer1 = 32
        layer2 = 64
        layer3 = 128
        layer4 = 256

        super().__init__()

        self.conv1 = nn.Conv2d(1, layer1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(layer1)
        self.conv2 = nn.Conv2d(layer1, layer2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(layer2)

        self.res23 = BasicBlock(layer2)

        self.conv3 = nn.Conv2d(layer2, layer3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(layer3)
        self.conv4 = nn.Conv2d(layer3, layer4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(layer4)


        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)

        self.dropout_fc = nn.Dropout(0.4)
        self.dropout_conv = nn.Dropout(0.25)

        self.fc1   = nn.Linear(layer4 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, 47)


    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.res23(x)
        x = self.pool(x) 

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x