import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import KMNIST
import matplotlib.pyplot as plt
from network import RBM
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load KMNIST dataset
trainset = KMNIST(root='./data', train=True, download=True, transform=transform)
testset = KMNIST(root='./data', train=False, download=True, transform=transform)

# Define DataLoader
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)


def train_rbm(model, train_loader, learning_rate=0.001, epochs=25, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, _ in train_loader:
            data = data.view(-1, 28 * 28)  # Flatten the images to 1D vector

            optimizer.zero_grad()
            positive_grad, negative_grad = model.contrastive_divergence(data)

            # Update weights using CD-5 gradient
            loss = torch.mean(torch.pow(data - model.sample_v(model.sample_h(data)), 2))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


model_16 = RBM(visible_units=28 * 28, hidden_units=16)
train_rbm(model_16, train_loader)

def plot_reconstruction(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(-1, 28 * 28)
            reconstructed = model(data)

            # Visualize original and reconstructed images
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(data[0].view(28, 28), cmap='gray')
            ax[0].set_title("Original")
            ax[1].imshow(reconstructed[0].view(28, 28), cmap='gray')
            ax[1].set_title("Reconstructed")
            plt.show()
            break  # Only show for the first batch


# Example for M=16
plot_reconstruction(model_16, test_loader)
