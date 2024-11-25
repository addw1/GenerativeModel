import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

class CVAE(nn.Module):
    def __init__(self, n_in, n_hid, z_dim, c_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_in + c_dim, n_hid)
        self.fc21 = nn.Linear(n_hid, z_dim) # z mean
        self.fc22 = nn.Linear(n_hid, z_dim) # z log variance
        self.fc3 = nn.Linear(z_dim + c_dim, n_hid)
        self.fc4 = nn.Linear(n_hid, n_in)

    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=-1)
        h1 = F.relu(self.fc1(x_c))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=-1)
        h3 = F.relu(self.fc3(z_c))
        reconstruction = torch.sigmoid(self.fc4(h3))
        return reconstruction

    def forward(self, x, c):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # ELBO = Eqϕ(z|x)[logpθ(x|z)] − DKL[qϕ(z|x)||p(z)]
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        DKL = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        return BCE + DKL

if __name__ == '__main__':
    batch_size = 128
    epochs = 20
    learning_rate = 1e-2
    n_in = 784  # Fashion MNIST image size (28x28 pixels)
    n_hid = 400  # Hidden layer size
    z_dim = 20  # Latent space dimension
    c_dim = 10  # Number of classes in Fashion MNIST (10 classes)


    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten images to 1D
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # create validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = CVAE(n_in, n_hid, z_dim, c_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # one hot
            # c = torch.zeros(target.size(0), c_dim).to(data.device)
            # c[torch.arange(target.size(0)), target] = 1
            c = one_hot(target, 10)
            optimizer.zero_grad()

            # forward
            recon_batch, mu, logvar = model(data, c)

            # loss
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()

            # update
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                #c = torch.zeros(target.size(0), c_dim).to(data.device)
                # one hot
                # c[torch.arange(target.size(0)), target] = 1
                c = one_hot(target, 10)
                recon_batch, mu, logvar = model(data, c)
                loss = model.loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")


    import matplotlib.pyplot as plt
    def plot_images(original, reconstructed, n=10):
        plt.figure(figsize=(10, 4))
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(original[i].view(28, 28).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i].view(28, 28).cpu().numpy(), cmap='gray')
            plt.axis('off')
        plt.show()

    # eval
    model.eval()
    test_loss = 0
    is_first = True
    with torch.no_grad():
        for data, target in test_loader:
            # c = torch.zeros(target.size(0), c_dim).to(data.device)
            # one hot
            # c[torch.arange(target.size(0)), target] = 1
            c = one_hot(target, 10)
            recon_batch, mu, logvar = model(data, c)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if is_first:
                is_first = False
                plot_images(data, recon_batch, n=10)

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # save pth
    torch.save(model.state_dict(), "cvae_model_dict.pth")