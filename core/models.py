import torch
import torch.nn as nn

from core.configs.values import VAEModelType, DatasetLabelType


class VanillaVAE(nn.Module):
    """
    Variational Autoencoder specifically designed for MedMNIST datasets
    """

    def __init__(self, img_shape, latent_dim=128, model_type: str | VAEModelType = VAEModelType.VAE, **kwargs):
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


class HybridConditionEncoder(nn.Module):
    """
    Encoder for hybrid conditioning in Conditional VAE

    This class supports the following features:
    - Encoding conditional information for different datasets;
    - Supporting single-label and multi-label;
    - Outputting a unified dimensional conditional vector unified_condition_dim.
    """

    def __init__(
            self,
            num_datasets: int,
            label_type_info: dict,
            dataset_embed_dim: int,
            class_embed_dim: int,
            unified_condition_dim: int = 32,
            **kwargs
    ):
        super().__init__()
        self.num_datasets = num_datasets
        self.label_type_info = label_type_info
        self.dataset_embed_dim = dataset_embed_dim
        self.class_embed_dim = class_embed_dim
        self.unified_condition_dim = unified_condition_dim

        self._dataset_embedding = nn.Embedding(num_datasets, dataset_embed_dim)

        # Label encoders by dataset
        self.single_label_encoders = nn.ModuleDict()
        self.multi_label_encoders = nn.ModuleDict()

        for dataset_name, info in label_type_info.items():
            if info['type'] == DatasetLabelType.MULTI:
                self.multi_label_encoders[dataset_name] = nn.Linear(info['n_classes'], class_embed_dim)
            else:
                self.single_label_encoders[dataset_name] = nn.Embedding(info['n_classes'], class_embed_dim)

        self.condition_projection = nn.Linear(dataset_embed_dim + class_embed_dim, unified_condition_dim)

    def forward(
            self,
            dataset_ids,
            dataset_names,
            label_types,
            single_labels=None,
            multi_labels=None,
            single_mask=None,
            multi_mask=None,
            **kwargs
    ):
        """
        @param dataset_ids: Tensor of dataset IDs (shape: [batch_size])
        """
        bsz = dataset_ids.size(0)
        device = dataset_ids.device

        # Get dataset embedding
        dataset_embedding = self._dataset_embedding(dataset_ids)

        # Initialize label embeddings
        label_embeddings = torch.zeros(bsz, self.class_embed_dim, device=device)

        # Process single-label datasets
        if single_mask is not None and single_mask.any() and single_labels is not None:
            self._process_dataset(
                mask=single_mask,
                labels=single_labels,
                encoder=self.single_label_encoders,
                dataset_names=dataset_names,
                embeddings=label_embeddings
            )

        # process multi-label datasets
        if multi_mask is not None and multi_mask.any() and multi_labels is not None:
            self._process_dataset(
                mask=multi_mask,
                labels=multi_labels,
                encoder=self.multi_label_encoders,
                dataset_names=dataset_names,
                embeddings=label_embeddings
            )

        # Concatenate dataset embedding and label embeddings
        condition_embedding = torch.cat([dataset_embedding, label_embeddings], dim=-1)  # [batch_size, dataset_embed_dim + class_embed_dim]

        # Project to unified condition dimension
        unified_condition = self.condition_projection(condition_embedding)  # [batch_size, unified_condition_dim]
        return unified_condition


    def _process_dataset(self, mask, labels, encoder, dataset_names, embeddings):
        """
        For all samples with the same format in a batch (single-label or multi-label):
        - convert the labels into embedding vectors using the corresponding embedding layer based on their respective dataset (single-label uses nn.Embedding, multi-label uses nn.Linear)
        - place them in the corresponding positions of class_embeds.

        @param mask: Boolean mask indicating which samples to process

        Example:
            >>> mask = torch.tensor([1, 0, 1, 0, 1])  #
            >>> labels = torch.tensor([1, 2, 0]) # Labels corresponding to the mask for the indices 1st, 3rd, and 5th samples
            >>> dataset_names = ['ds1', 'ds2', 'ds3', 'ds2', 'ds1']  # assuming `ds1`, `ds3` are single-label datasets and `ds2` is a multi-label dataset


        """
        device = mask.device
        encoder = encoder.to(device)
        if mask.any():
            indices = mask.nonzero().squeeze(-1).to(device)  # the output is [0, 2, 4] for the example above
            for idx in indices:
                dataset_name = dataset_names[idx]
                label = labels[mask[:idx.item() + 1].sum() - 1]  # Get the label corresponding to the current index
                embeddings[idx] = encoder[dataset_name](label)  # Use the appropriate encoder for the dataset


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
            is_multi_label: bool = False,
            num_datasets: int = 1,
            label_type_info: dict = None,
            dataset_embed_dim: int = 32,
            class_embed_dim: int = 32,
            use_hybrid_conditioning: bool = False,
            **kwargs
    ):
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.is_multi_label = is_multi_label
        self.num_datasets = num_datasets
        self.label_type_info = label_type_info
        self.dataset_embed_dim = dataset_embed_dim
        self.class_embed_dim = class_embed_dim
        self.use_hybrid_conditioning = use_hybrid_conditioning

        super().__init__(img_shape=img_shape, latent_dim=latent_dim, model_type=model_type, **kwargs)

    def _init_model_architecture(self):
        latent_dim = self.latent_dim
        c, h, w = self.img_shape

        if self.use_hybrid_conditioning and self.label_type_info:
            self.condition_encoder = HybridConditionEncoder(
                num_datasets=self.num_datasets,
                label_type_info=self.label_type_info,
                dataset_embed_dim=self.dataset_embed_dim,
                class_embed_dim=self.class_embed_dim,
                unified_condition_dim=self.condition_dim
            )
            total_condition_dim = self.condition_dim
        else:
            # Condition embedding
            if self.is_multi_label:
                self.condition_embedding = nn.Linear(self.num_classes, self.condition_dim)
            else:
                self.condition_embedding = nn.Embedding(self.num_classes,
                                                        self.condition_dim)  # Out: [batch_size, condition_dim]

            total_condition_dim = self.condition_dim


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
        self.fc_mu = nn.Linear(self.flattened_size + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size + total_condition_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + total_condition_dim, self.flattened_size)
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

    def _get_condition_embedding(self, **kwargs):
        """
        Get the condition embedding based on the labels.
        If multi-label, use linear embedding; otherwise, use embedding layer.
        """
        if self.use_hybrid_conditioning and hasattr(self, 'condition_encoder'):
            # Use hybrid condition encoder
            return self.condition_encoder(**kwargs)

        # Fallback to standard condition embedding
        labels = kwargs['labels']
        if self.is_multi_label:
            return self.condition_embedding(labels.float())
        else:
            return self.condition_embedding(labels)

    def encode(self, x, **condition_kwargs):
        """Encode input to latent parameters with condition"""
        condition_embed = self._get_condition_embedding(**condition_kwargs)
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

    def decode(self, z, **condition_kwargs):
        condition_embed = self._get_condition_embedding(**condition_kwargs)

        """Decode latent variables to reconstruction with condition"""
        z_cond = torch.cat([z, condition_embed], dim=1)

        # Decode the latent variables
        h = self.fc_decode(z_cond)
        h = h.view(-1, *self._final_feature_shape)
        return self.decoder(h)

    def forward(self, x, **condition_kwargs):
        """Forward pass through Conditional VAE"""
        mu, logvar = self.encode(x, **condition_kwargs)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, **condition_kwargs)
        return x_recon, mu, logvar


def get_model(model_name: str):
    if model_name == VAEModelType.VAE:
        return VanillaVAE
    elif model_name == VAEModelType.CVAE:
        return ConditionalVAE
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported types are: {VAEModelType.__members__.keys()}")
