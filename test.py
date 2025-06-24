import sys
sys.path.insert(0, "../")

from typing import Tuple

from core.core import Core
from core.loss_function import LossFunction
from core.optimizer import Optimizer
from core.vae_agent import VariationalAutoEncoder

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from core.utils.general import set_random_seed, root_path
from core.configs.arguments import get_arguments

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


class Model(nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )


    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 64, 7, 7)
        x_hat = self.decoder(h)
        return x_hat
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x_hat, mu, logvar, x):
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld


if __name__ == '__main__':
    args = get_arguments()
    set_random_seed(args.seed)
    root = root_path()

    device = torch.device(args.device)
    print(device)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./datasets/training", train=True, transform=transform, download=True)

    network = Model(img_size=(28, 28), latent_dim=400)
    loss_module = VAELoss()

    agent = VariationalAutoEncoder(model=network, device=device)
    optimizer = Optimizer(optimizer=args.optimizer,
                          model_parameters=agent.get_parameters(),
                          config={'lr': args.learning_rate, "weight_decay": args.weight_decay})
    loss_function = LossFunction(loss_function=loss_module,
                                 device=device)
    core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function)

    training_metrics, _ = core.train(training_data=dataset, batch_size=args.batch_size, epochs=args.epochs)

    with torch.no_grad():
        num_samples = 16
        noise = torch.randn(num_samples, 400).to(device=device)
        images = network.decode(noise)
        for i in range(num_samples):
            save_image(images.view(num_samples, 1, 28, 28), 'generated_sample.png')

