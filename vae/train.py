import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from params import generate_parser
from network import VAE

n_in = 784
n_hid = 400
z_dim = 20
learning_rate = 1e-3
batch_size = 128
epochs = 10


# --- defines the loss function --- #
def loss_function(recon_x, x, mu, logvar):
    # ELBO = Eqϕ(z|x)[logpθ(x|z)] − DKL[qϕ(z|x)||p(z)]
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    DKL = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return BCE + DKL

# create data set
train_data = datasets.FashionMNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.FashionMNIST('./data', train=False,
                           transform=transforms.ToTensor())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size, shuffle=True)

# --- train and test --- #
model = VAE(n_in=n_in,n_hid=n_hid,z_dim=z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
log_interval = 100
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # data: [batch size, 1, 28, 28]
        # label: [batch size] -> we don't use
        optimizer.zero_grad()
        data = data.to(device)
        recon_data, mu, logvar = model(data)
        loss = loss_function(recon_data, data, mu, logvar)
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            cur_loss = loss_function(recon_data, data, mu, logvar).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(batch_size, 8)
                comparison = torch.cat(
                    [data[:num_samples], recon_data.view(batch_size, 1, 28, 28)[:num_samples]]).cpu()
                save_generated_img(
                    comparison, 'reconstruction', epoch, num_samples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# --- etc. funtions --- #
import os
def save_generated_img(image, name, epoch, nrow=8):
    if not os.path.exists('results'):
        os.makedirs('results')

    if epoch % 5 == 0:
        save_path = 'results/'+name+'_'+str(epoch)+'.png'
        save_image(image, save_path, nrow=nrow)


def sample_from_model(epoch):
    with torch.no_grad():
        # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
        sample = torch.randn(64, z_dim).to(device)
        sample = model.decode(sample).cpu().view(64, 1, 28, 28)
        save_generated_img(sample, 'sample', epoch)


def plot_along_axis(epoch):
    z1 = torch.arange(-2, 2, 0.2).to(device)
    z2 = torch.arange(-2, 2, 0.2).to(device)
    num_z1 = z1.shape[0]
    num_z2 = z2.shape[0]
    num_z = num_z1 * num_z2

    sample = torch.zeros(num_z, 20).to(device)

    for i in range(num_z1):
        for j in range(num_z2):
            idx = i * num_z2 + j
            sample[idx][0] = z1[i]
            sample[idx][1] = z2[j]

    sample = model.decode(sample).cpu().view(num_z, 1, 28, 28)
    save_generated_img(sample, 'plot_along_z1_and_z2_axis', epoch, num_z1)


# --- main function --- #
if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        sample_from_model(epoch)
        plot_along_axis(epoch)