import torch
import torch.nn as nn

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
        return torch.bernoulli(p)

    def sample_gaussian(self, p):
        return torch.normal(p, torch.ones_like(p))

    def P_h_x(self, x):
        """Returns the conditional P(h|x)."""
        activation = torch.matmul(x, self.W.t()) + self.b
        return torch.sigmoid(activation)

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
        h_sample = self.sample(h_prob)
        x_k = x
        for _ in range(self.k):
            h_k = self.sample(self.P_h_x(x_k))
            x_k = self.sample_gaussian(self.P_x_h(h_k))
        return x_k, self.P_x_h(h_k)