import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import transforms

class VAE(nn.Module):
    def __init__(self, n_in, n_hid, z_dim, c_dim):
        super().__init__()
        assert c_dim == 0
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc21 = nn.Linear(n_hid, z_dim) # z mean
        self.fc22 = nn.Linear(n_hid, z_dim) # z log variance
        self.fc3 = nn.Linear(z_dim, n_hid)
        self.fc4 = nn.Linear(n_hid, n_in)

    def encode(self, x, c):
        assert c == None
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        # avoid negative value
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Implements: z = mu + epsilon*stdev."""
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z, c):
        assert c == None
        h3 = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h3))
        return reconstruction

    def forward(self, x, c):
        assert c == None
        x = x.view(-1, 784)
        # enocode for z u and var
        mu, logvar = self.encode(x, c)
        # generate z
        z = self.reparameterize(mu, logvar)
        # decode z
        return self.decode(z, c), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # ELBO = Eqϕ(z|x)[logpθ(x|z)] − DKL[qϕ(z|x)||p(z)]
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        DKL = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        return BCE + DKL

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 128
    epochs = 20
    learning_rate = 1e-3
    n_in = 784
    n_hid = 400
    z_dim = 20
    c_dim = 0

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


    model = VAE(n_in, n_hid, z_dim, c_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # one hot
            c = None
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
                c = None
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
            c = None
            recon_batch, mu, logvar = model(data, c)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if is_first:
                is_first = False
                plot_images(data, recon_batch, n=10)

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # save pth
    torch.save(model.state_dict(), "ave_model_dict.pth")