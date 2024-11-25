import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from network import VAE
from cvae import CVAE, one_hot
from sklearn.manifold import TSNE

# load dataset
test_dataset = datasets.FashionMNIST(
    root="data", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# load model
n_in = 784
n_hid = 400
z_dim = 20

vae = VAE(n_in=784, n_hid=400, z_dim=20,c_dim=0)
cvae = CVAE(n_in=784, n_hid=400, z_dim=20, c_dim=10)
vae.load_state_dict(torch.load("ave_model_dict.pth"))
cvae.load_state_dict(torch.load("cvae_model_dict.pth"))

vae.eval()
cvae.eval()

vae_latents, cvae_latents, labels = [], [], []

# get u
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(data.size(0), -1)
        # c = torch.zeros(target.size(0), 10).to(data.device)
        # c[torch.arange(target.size(0)), target] = 1
        c = one_hot(target, 10)
        mu_vae, _ = vae.encode(data, None)
        mu_cvae, _ = cvae.encode(data, c)

        vae_latents.append(mu_vae)
        cvae_latents.append(mu_cvae)
        labels.append(target)

vae_latents = torch.cat(vae_latents).numpy()
cvae_latents = torch.cat(cvae_latents).numpy()
labels = torch.cat(labels).numpy()

tsne = TSNE(n_components=2, random_state=63)
vae_2d = tsne.fit_transform(vae_latents)
cvae_2d = tsne.fit_transform(cvae_latents)

# set class name
class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for label in range(10):
    indices = labels == label
    plt.scatter(vae_2d[indices, 0], vae_2d[indices, 1], s=10, label=class_names[label])
plt.title("VAE: Latent Space Visualization (t-SNE)")
plt.legend(fontsize=8)
plt.grid()

plt.subplot(1, 2, 2)
for label in range(10):
    indices = labels == label
    plt.scatter(cvae_2d[indices, 0], cvae_2d[indices, 1], s=10, label=class_names[label])
plt.title("c-VAE: Latent Space Visualization (t-SNE)")
plt.legend(fontsize=8)
plt.grid()

plt.tight_layout()
plt.show()
