import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import torch
from network import VAE
from cvae import CVAE
# 假设你的VAE和C-VAE已经被训练并且可以被导入
# from your_model import VAE, C_VAE
# vae = VAE()
# c_vae = C_VAE()
vae = VAE(n_in=784, n_hid=400, z_dim=20,c_dim=0)
c_vae = CVAE(n_in=784, n_hid=400, z_dim=20,c_dim=10)
vae.load_state_dict(torch.load("ave_model_dict.pth"))
c_vae.load_state_dict(torch.load("cvae_model_dict.pth"))

# 加载Fashion MNIST测试数据集
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 1. 通过编码器传递测试数据集的样本并保留均值
vae_mu = []
c_vae_mu = []
labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # VAE的编码器输出
        data = data.view(data.size(0), -1)
        c = torch.zeros(target.size(0), 10).to(data.device)
        c[torch.arange(target.size(0)), target] = 1
        vae_z, _ = vae.encode(data, None)
        # C-VAE的编码器输出
        c_vae_z, _ = c_vae.encode(data, c)
        vae_mu.append(vae_z)
        c_vae_mu.append(c_vae_z)
        labels.append(target)

vae_mu = torch.cat(vae_mu).numpy()
c_vae_mu = torch.cat(c_vae_mu).numpy()
labels = torch.cat(labels).numpy()



# 2. 使用TSNE将均值映射到2D流形
tsne = TSNE(n_components=2, random_state=0)
vae_mu_2d = tsne.fit_transform(vae_mu)
c_vae_mu_2d = tsne.fit_transform(c_vae_mu)

# 3. 绘制2D流形，并按类别着色
def plot_tsne(X, labels, title):
    plt.figure(figsize=(8, 6))
    for i in range(10):
        idx = np.where(labels == i)
        plt.scatter(X[idx, 0], X[idx, 1], label=str(i), alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.show()

plot_tsne(vae_mu_2d, labels, 'VAE Manifold')
plot_tsne(c_vae_mu_2d, labels, 'C-VAE Manifold')

# 4. 比较两个流形上的数据分布，并描述你的观察结果
# 这里你需要根据上面的绘图结果来描述你的观察

# 5. 提出一个假设来解释你的观察现象
# 这里你需要根据你的观察提出一个可能的解释