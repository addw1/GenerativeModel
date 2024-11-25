import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from network import Generator, Discriminator
# prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
g_optim = optim.Adam(gen.parameters(), lr=1e-4)
d_optim = optim.Adam(dis.parameters(), lr=1e-5)
loss_fun = torch.nn.MSELoss()



# 训练GAN
G_loss = []
D_loss = []
for epoch in range(10):
    g_epoch_loss = 0
    d_epoch_loss = 0
    count = len(train_dataloader)
    for step, (img, _) in enumerate(train_dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)
        d_optim.zero_grad()
        real_output = dis(img)
        real_loss = loss_fun(real_output, torch.ones_like(real_output, device=device))
        real_loss.backward()
        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())
        fake_loss = loss_fun(fake_output, torch.zeros_like(fake_output, device=device))
        fake_loss.backward()

        d_loss = real_loss + fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fun(fake_output, torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print("Epoch:", epoch)

def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2, cmap="gray")
        plt.axis("off")
    plt.show()


test_input = torch.randn(16, 100, device=device)
gen_img_plot(gen, test_input)


torch.save(dis.state_dict(), 'model_state_dict.pth')