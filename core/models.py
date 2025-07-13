import torch
import torch.nn as nn

from core.configs.values import VAEModelType


class VanillaVAE(nn.Module):
    """
    Variational Autoencoder specifically designed for MedMNIST datasets
    """

    def __init__(self, img_shape, latent_dim=128,  model_type: str | VAEModelType = VAEModelType.VAE, **kwargs):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.model_type = model_type

        self._init_model_architecture()

    def _init_model_architecture(self):
        latent_dim = self.latent_dim
        c, h, w = self.img_shape  # channels, height, width. e.g., (1, 28, 28) for grayscale images

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.encoder(dummy)
            self.flattened_size = int(out.view(1, -1).size(1))
            self._final_feature_shape = out.shape[1:]  # (128, 4, 4) for example

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),  # [B, c, 32, 32]
            nn.AdaptiveAvgPool2d((h, w)),  # Ensure correct output size
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variables to reconstruction"""
        h = self.fc_decode(z)
        h = h.view(-1, *self._final_feature_shape)
        return self.decoder(h)

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class ConditionalVAE(VanillaVAE):
    """
    Conditional VAE for MedMNIST datasets
    """

    def __init__(
            self,
            img_shape,
            latent_dim=128,
            model_type: str | VAEModelType = VAEModelType.CVAE,
            num_classes=10,
            condition_dim: int = 32,
            **kwargs
    ):
        self.num_classes = num_classes
        self.condition_dim = condition_dim

        super().__init__(img_shape=img_shape, latent_dim=latent_dim, model_type=model_type, **kwargs)

    def _init_model_architecture(self):
        latent_dim = self.latent_dim
        c, h, w = self.img_shape

        # Condition embedding
        self.condition_embedding = nn.Embedding(self.num_classes,
                                                self.condition_dim)  # Out: [batch_size, condition_dim]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c + 1, 32, kernel_size=4, stride=2, padding=1),
            # c+1 for a condition channel, out: [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy_img = torch.zeros(1, c, h, w)
            dummy_cond = torch.zeros(1, 1, h, w)  # Condition channel
            dummy_input = torch.cat([dummy_img, dummy_cond], dim=1)
            out = self.encoder(dummy_input)
            self.flattened_size = int(out.view(1, -1).size(1))
            self._final_feature_shape = out.shape[1:]

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size + self.condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size + self.condition_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + self.condition_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),  # [B, c, 32, 32]
            nn.AdaptiveAvgPool2d((h, w)),  # Ensure correct output size
            nn.Sigmoid()
        )

    def _transform_condition_embed_to_channel(self, condition_embed):
        """
        Transform the condition embedding to a channel that can be concatenated with the input image.
        """
        # Reshape to match the image dimensions
        assert condition_embed.dim() == 2, f"Condition embedding should be of shape [batch_size, condition_dim], got {condition_embed.shape}"
        batch_size = condition_embed.shape[0]
        _, h, w = self.img_shape

        # Take mean across the condition embedding to create a single channel
        condition_channel = condition_embed.mean(dim=1, keepdim=True)  # [batch_size, 1]
        condition_channel = condition_channel.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1, 1]

        # Expand to match the image dimensions
        condition_channel = condition_channel.expand(batch_size, 1, h, w)  # [batch_size, 1, h, w]
        return condition_channel


    def encode(self, x, labels):
        """Encode input to latent parameters with condition"""
        condition_embed = self.condition_embedding(labels)
        condition_channel = self._transform_condition_embed_to_channel(condition_embed)

        # Concatenate condition channel with input image
        x_cond = torch.cat([x, condition_channel], dim=1)
        h = self.encoder(x_cond)

        # Flatten the output
        h = h.view(h.size(0), -1)

        # Concatenate with condition embedding
        h_cond = torch.cat([h, condition_embed], dim=1)
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar

    def decode(self, z, labels):
        condition_embed = self.condition_embedding(labels)

        """Decode latent variables to reconstruction with condition"""
        z_cond = torch.cat([z, condition_embed], dim=1)

        # Decode the latent variables
        h = self.fc_decode(z_cond)
        h = h.view(-1, *self._final_feature_shape)
        return self.decoder(h)

    def forward(self, x, labels):
        """Forward pass through Conditional VAE"""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar


def get_model(model_name: str):
    if model_name == VAEModelType.VAE:
        return VanillaVAE
    elif model_name == VAEModelType.CVAE:
        return ConditionalVAE
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported types are: {VAEModelType.__members__.keys()}")
