import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np

# %%
device = 'cpu'
batch_size = 128
transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Resize(16)
     ])

train_set = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# %%
# Defining Gaussian-Bernoulli RBM
class RBM(nn.Module):
    def __init__(self, D: int, F: int, k: int):
        """
        Args:
        D: Size of the input data.
        F: Size of the hidden variable.
        k: Number of MCMC iterations for negative sampling.

        The function initializes the weight (W) and biases (c & b).
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(F, D) * 1e-2)  # Initialized from Normal(mean=0.0, variance=1e-4)
        self.c = nn.Parameter(torch.zeros(D))  # Initialized as 0.0
        self.b = nn.Parameter(torch.zeros(F))  # Initilaized as 0.0
        self.k = k

    def sample(self, p):
        #p = torch.clamp(p, 0, 1)
        return torch.bernoulli(p)

    def sample_gaussian(self, p):
        # p = torch.clamp(p, 0, 1)
        return torch.normal(p, torch.ones_like(p))

    def P_h_x(self, x):
        """Returns the conditional P(h|x)."""
        activation = torch.matmul(x, self.W.t()) + self.b
        activation = torch.sigmoid(activation)
        return activation
    def P_x_h(self, h):
        activation = torch.matmul(h, self.W) + self.c
        return activation

    def free_energy(self, x):
        """Returns the Average F(x) free energy. (Slide 11)."""
        wx_b = torch.matmul(x, self.W.t()) + self.b
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        vbias_term = torch.matmul(x, self.c)
        return -hidden_term - vbias_term

    def forward(self, x):
        """Generates x_negative using MCMC Gibbs sampling starting from x."""
        # Positive phase
        h_prob = self.P_h_x(x)
        # h_sample = self.sample(h_prob)
        x_k = x
        for _ in range(self.k):
            h_k = self.sample(self.P_h_x(x_k))
            x_k = self.sample_gaussian(self.P_x_h(h_k))

        return x_k, self.P_x_h(h_k)


def train(model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # flatten and pre-process variable
        data = data.view(data.size(0), -1).to(device)
        mean, std = data.mean(), data.std()
        data = (data - mean) / (std + 1e-8)
        optimizer.zero_grad()

        x_neg, _ = model(data)
        loss = torch.mean(model.free_energy(data) - model.free_energy(x_neg))

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % (len(train_loader) // 2) == 0:
            print('Train({})[{:.0f}%]: Loss: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), train_loss / (batch_idx + 1)))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)

            mean, std = data.mean(), data.std()
            data = (data - mean) / std

            x_neg, _ = model(data)
            test_loss += torch.mean((data - x_neg) ** 2).item()

    test_loss /= len(test_loader)
    print('Test({}): Loss: {:.4f}'.format(epoch, test_loss))


def show(img1, img2):
    npimg1 = img1.cpu().numpy()
    npimg2 = img2.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(np.transpose(npimg1, (1, 2, 0)), interpolation='nearest')
    axes[1].imshow(np.transpose(npimg2, (1, 2, 0)), interpolation='nearest')
    fig.show()




seed = 42
num_epochs = 25
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
M = [16]

rbm = RBM(D=28 * 28 * 1, F=M[0], k=5).to(device)
# Keep the lr small to avoid overflow
optimizer = optim.Adam(rbm.parameters(), lr=1e-4)
print("M = {}:".format(M[0]))
for epoch in range(1, num_epochs + 1):
    train(rbm, device, train_loader, optimizer, epoch)
    test(rbm, device, test_loader, epoch)
    # reconstructing samples for plotting
    data, _ = next(iter(test_loader))
    data = data[:32]
    data_size = data.size()
    data = data.view(data.size(0), -1).to(device)
    mean, std = data.mean(), data.std()
    bdata = (data - mean) / std
    vh_k, pvh_k = rbm(bdata)
    vh_k, pvh_k = vh_k.detach(), pvh_k.detach()

    show(make_grid(data.reshape(data_size), padding=0),
         make_grid(pvh_k.reshape(data_size).clip(min=0, max=1), padding=0))
    plt.show()

    # from train set
    # reconstructing samples for plotting
    data, _ = next(iter(train_loader))
    data = data[:32]
    data_size = data.size()
    data = data.view(data.size(0), -1).to(device)
    mean, std = data.mean(), data.std()
    bdata = (data - mean) / std
    vh_k, pvh_k_train = rbm(bdata)
    vh_k, pvh_k_train = vh_k.detach(), pvh_k_train.detach()

    show(make_grid(data.reshape(data_size), padding=0),
         make_grid(pvh_k_train.reshape(data_size).clip(min=0, max=1), padding=0))
    plt.show()

    print('Optimizer Learning rate: {0:.4f}\n'.format(optimizer.param_groups[0]['lr']))