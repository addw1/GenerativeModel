import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(256 * 7 * 7)
        # i=(o−1)∗s+k−2∗p
        self.decon1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 4),
                                         stride=2,
                                         padding=1)  # (128, 14, 14)
        self.bn2 = nn.BatchNorm2d(128)
        self.decon2 = nn.ConvTranspose2d(128, 64,
                                         kernel_size=(4, 4),
                                         stride=2,
                                         padding=1)  # (64, 28, 28)
        self.bn3 = nn.BatchNorm2d(64)
        self.decon3 = nn.ConvTranspose2d(64, 32,
                                         kernel_size=(3, 3),
                                         stride=1,
                                         padding=1)  # (32, 28, 28)
        self.bn4 = nn.BatchNorm2d(32)
        self.decon4 = nn.ConvTranspose2d(32, 1,
                                         kernel_size=(3, 3),
                                         stride=1,
                                         padding=1)  # (1, 28, 28)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        # (batch_size, 256, 7, 7)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.decon1(x))
        x = self.bn2(x)
        x = F.relu(self.decon2(x))
        x = self.bn3(x)
        x = torch.tanh(self.decon3(x))
        x = self.bn4(x)
        x = torch.tanh(self.decon4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # o = (i - k + 2*p) / s + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1) #14*14*64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2)# 7 * 7 * 128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)# 4 * 4 * 256
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # 4 * 4 * 512
        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = F.dropout2d(F.leaky_relu_(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu_(self.conv2(x)))
        x = F.dropout2d(F.leaky_relu_(self.conv3(x)))
        x = F.dropout2d(F.leaky_relu_(self.conv4(x)))
        x = self.bn(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc(x)
        return x