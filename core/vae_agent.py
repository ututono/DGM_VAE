import logging
from dataclasses import dataclass, asdict
from os import PathLike
from pathlib import Path

from core.agent import AbstractAgent
import torch

from torch.utils.data import DataLoader

from core.configs.values import VAEModelType, DatasetLabelType, DatasetLabelInfoNames
from core.models import get_model
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.utils.metrics import Metrics
from typing import List, Dict, Any


@dataclass
class MonoConditionConfig:
    labels: torch.Tensor = None
    use_hybrid_conditioning: bool = False


@dataclass
class HybridConditionConfig:
    dataset_ids: torch.Tensor
    dataset_names: List[str]
    label_types: List[str]
    single_labels: torch.Tensor = None
    multi_labels: torch.Tensor = None
    single_mask: torch.Tensor = None
    multi_mask: torch.Tensor = None


@dataclass
class ConditionConfig:
    condition_config: MonoConditionConfig | HybridConditionConfig = None
    use_hybrid_conditioning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self.condition_config)
        result['use_hybrid_conditioning'] = self.use_hybrid_conditioning
        return result


TYPE_NAME = DatasetLabelInfoNames.TYPE.value
N_CLASS_NAME = DatasetLabelInfoNames.N_CLASSES.value

logger = logging.getLogger(__name__)

class VariationalAutoEncoder(AbstractAgent):
    def __init__(self, model, device):
        train_metrics = Metrics(["loss"])  # TODO add more metrics like KL divergence, reconstruction loss
        val_metrics = Metrics(["loss"])
        test_metrics = Metrics(["loss"])

        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            # 'test': test_metrics
        }
        self.records_manager = None  # Will be set in save_checkpoint/load_checkpoint methods
        super().__init__(model=model, device=device, metrics=metrics)

    def _perform_training_epoch(self, epoch, train_dataloader: DataLoader,
                                optimizer: Optimizer,
                                loss_fn: LossFunction):

        epoch_losses = list()
        total_samples = 0

        for batch_data in train_dataloader:
            if self.is_conditional_training:
                # Handle hybrid multi-dataset conditioned samples
                if isinstance(batch_data, dict):
                    x = batch_data['images'].to(self._device)
                    condition_kwargs = self._prepare_condition_kwargs(batch_data)
                    x_hat, mu, logvar = self._model(x, **condition_kwargs)
                else:
                    # Legacy single dataset format
                    x, y = batch_data
                    x = x.to(self._device)
                    y = y.to(self._device).squeeze(dim=1)
                    x_hat, mu, logvar = self._model(x, labels=y)
            else:
                # For Vanilla VAE, we only need the images
                x = batch_data['images'].to(self._device) if isinstance(batch_data, dict) else batch_data[0].to(
                    self._device)
                x_hat, mu, logvar = self._model(x)

            loss: torch.Tensor = loss_fn(x_hat, mu, logvar, x)
            epoch_losses.append(loss.item())
            total_samples += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss: float = sum(epoch_losses) / total_samples
        self._metrics['train'].update(epoch=epoch, batch_loss=epoch_loss)

    def _perform_evaluation_epoch(self, epoch, eval_dataloader: DataLoader, loss_fn: LossFunction):

        epoch_losses_val = list()
        total_samples = 0

        with torch.no_grad():
            for batch_data in eval_dataloader:  # x: (images, labels)
                if self.is_conditional_training:
                    if isinstance(batch_data, dict):
                        x = batch_data['images'].to(self._device)
                        condition_kwargs = self._prepare_condition_kwargs(batch_data)
                        x_hat, mu, logvar = self._model(x, **condition_kwargs)
                    else:
                        x, y = batch_data
                        x = x.to(self._device)
                        y = y.to(self._device).squeeze(dim=1)
                        x_hat, mu, logvar = self._model(x, labels=y)
                else:
                    x = batch_data['images'].to(self._device) if isinstance(batch_data, dict) else batch_data[0].to(
                        self._device)
                    x_hat, mu, logvar = self._model(x)

                loss: torch.Tensor = loss_fn(x_hat, mu, logvar, x)
                epoch_losses_val.append(loss.item())
                total_samples += x.size(0)

        epoch_loss_validation: float = sum(epoch_losses_val) / total_samples
        self._metrics['validation'].update(epoch=epoch, batch_loss=epoch_loss_validation)

        # self._metrics['validation'].update(epoch=epoch, batch_loss=epoch_loss_validation)

    def _prepare_condition_kwargs(self, batch_data):
        """Prepare keyword arguments for the model based on the batch data."""
        if self.use_hybrid_conditioning:
            condition_config = HybridConditionConfig(
                dataset_ids=batch_data['dataset_ids'].to(self._device),
                dataset_names=batch_data['dataset_names'],
                label_types=batch_data['label_types'],
                single_labels=batch_data.get('single_labels', None),
                multi_labels=batch_data.get('multi_labels', None),
                single_mask=batch_data.get('single_mask', None),
                multi_mask=batch_data.get('multi_mask', None)
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
        else:
            condition_config = MonoConditionConfig(
                labels=self._process_labels(batch_data['labels']).to(
                    self._device) if self.is_conditional_training else None,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )

        return condition_kwargs.to_dict()

    @staticmethod
    def _process_labels(labels):
        """Process different dimensions of labels for Conditional VAE."""
        # handle 3D labels e.g., (batch_size, num_samples, num_classes)
        if len(labels.shape) == 3:
            # if the number of samples is 1, we can squeeze it
            if labels.shape[1] == 1:
                labels = labels.squeeze(dim=1)
            else:
                # Flatten the labels to (batch_size, num_classes)
                bsz = labels.shape[0]
                labels = labels.view(bsz, -1)
        # handle 2D labels, either single-label or multi-label e.g., (batch_size, num_classes)
        elif len(labels.shape) == 2:
            if labels.shape[1] == 1:
                # Single-label case, squeeze to (batch_size,)
                labels = labels.squeeze(dim=1)

        # no need to process 1D labels, they are already in the correct shape
        return labels

    def test(self, test_data):
        """Test the model on test data"""
        self._model.eval()
        test_losses = []
        total_samples = 0

        n_samples = 8
        comparison = None

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_data):
                if self.is_conditional_training:
                    if isinstance(batch_data, dict):
                        x = batch_data['images'].to(self._device)
                        condition_kwargs = self._prepare_condition_kwargs(batch_data)
                        x_hat, mu, logvar = self._model(x, **condition_kwargs)
                    else:
                        x, y = batch_data
                        x = x.to(self._device)
                        y = y.to(self._device).squeeze(dim=1)
                        x_hat, mu, logvar = self._model(x, labels=y)
                else:
                    x = batch_data['images'].to(self._device) if isinstance(batch_data, dict) else batch_data[0].to(
                        self._device)
                    x_hat, mu, logvar = self._model(x)

                if batch_idx == 0:
                    comparison = torch.cat([x[:n_samples], x_hat[:n_samples]])

                loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
                test_losses.append(loss.item())
                total_samples += x.size(0)

        average_test_loss = sum(test_losses) / total_samples
        results = {
            'test_loss(recon_loss)': average_test_loss,
            'comparison': comparison
        }
        return results

    def predict(
            self,
            num_samples: int = 1,
            labels=None,
            dataset_id: int = None,
            dataset_name: str = None,
            label_type: str = 'single',
            **kwargs
    ) -> torch.Tensor:
        self._model.eval()
        latent_dim = self._model.latent_dim if hasattr(self._model, 'latent_dim') else 128

        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, latent_dim).to(self._device)

            if self.is_conditional_training:
                if self.use_hybrid_conditioning:
                    generated_samples = self._generate_samples_hybrid_conditioning(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                        label_type=label_type,
                        labels=labels,
                        num_samples=num_samples,
                        z=z
                    )
                else:
                    generated_samples = self._generate_samples_mono_conditioning(
                        labels=labels,
                        num_samples=num_samples,
                        z=z
                    )
            else:
                # Generate samples using the decoder
                generated_samples = self._model.decode(z)

        return generated_samples

    def _generate_samples_mono_conditioning(self, labels, num_samples, z):
        # Prepare labels for Mono Conditional VAE
        if labels is None:
            # Generate labels based on model type
            if self._model.is_multi_label:
                # For multi-label, generate random binary vectors
                labels = torch.randint(0, 2, (num_samples, self._model.num_classes)).float().to(self._device)
            else:
                # For single-label, generate random class indices
                labels = torch.randint(0, self._model.num_classes, (num_samples,)).to(self._device)
        elif isinstance(labels, int):
            # Convert single integer to appropriate format
            if self._model.is_multi_label:
                # Create one-hot vector for multi-label model
                one_hot = torch.zeros(num_samples, self._model.num_classes)
                one_hot[:, labels] = 1.0
                labels = one_hot.to(self._device)
            else:
                # Repeat for single-label model
                labels = torch.tensor([labels] * num_samples).to(self._device)
        generated_samples = self._model.decode(z, labels)
        return generated_samples

    def _generate_samples_hybrid_conditioning(self, dataset_id, dataset_name, label_type, labels, num_samples, z):
        if dataset_id is None or dataset_name is None or label_type is None:
            raise ValueError("For hybrid conditioning, dataset_id, dataset_name, and label_type must be provided.")
        if labels is None:
            # Generate random labels based on label_type
            if label_type == DatasetLabelType.SINGLE:
                max_classes = self._model.label_type_info[dataset_name][N_CLASS_NAME]
                labels = torch.randint(0, max_classes, (num_samples,)).to(self._device)
            else:  # multi-label
                n_classes = self._model.label_type_info[dataset_name][N_CLASS_NAME]
                labels = torch.randint(0, 2, (num_samples, n_classes)).float().to(self._device)
        else:
            if isinstance(labels, int) and label_type == DatasetLabelType.SINGLE:
                # Convert single integer to appropriate format
                labels = torch.tensor([labels] * num_samples).to(self._device)

        condition_kwargs = self._prepare_condition_kwargs_for_generation(
            num_samples=num_samples,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            labels=labels,
            label_type=str(label_type)
        )
        generated_samples = self._model.decode(z, **condition_kwargs)
        return generated_samples

    def _prepare_condition_kwargs_for_generation(self, num_samples: int, dataset_id: int, dataset_name: str, labels,
                                                 label_type: str):
        dataset_ids = torch.tensor([dataset_id] * num_samples).to(self._device)
        dataset_names = [dataset_name] * num_samples
        label_types = [label_type] * num_samples

        if label_type == DatasetLabelType.SINGLE:
            condition_config = HybridConditionConfig(
                dataset_ids=dataset_ids,
                dataset_names=dataset_names,
                label_types=label_types,
                single_labels=labels,
                multi_labels=None,
                single_mask=torch.ones(num_samples, dtype=torch.bool),
                multi_mask=torch.zeros(num_samples, dtype=torch.bool)
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
        elif label_type == DatasetLabelType.MULTI:
            condition_config = HybridConditionConfig(
                dataset_ids=dataset_ids,
                dataset_names=dataset_names,
                label_types=label_types,
                single_labels=None,
                multi_labels=labels,
                single_mask=torch.zeros(num_samples, dtype=torch.bool),
                multi_mask=torch.ones(num_samples, dtype=torch.bool)
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
        else:
            raise ValueError(f"Unsupported label type: {label_type}. Supported types are 'single' and 'multi'.")
        return condition_kwargs.to_dict()

    def generate_conditional_samples(self, num_samples: int = 1, labels=None) -> torch.Tensor:
        """Generate samples for each specified label. Work for Conditional VAE only."""
        if not self.is_conditional_training:
            raise ValueError("This method is only applicable for Conditional VAE models.")

        self._model.eval()
        latent_dim = self._model.latent_dim
        generated_samples = []

        with torch.no_grad():
            # Sample from standard normal distribution
            for label in labels:
                samples = self.predict(num_samples=num_samples, labels=label)
                generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)

    def _prepare_generation_kwargs(self, num_samples, dataset_id, dataset_name, labels, label_type):
        """Prepare kwargs for sample generation"""
        dataset_ids = torch.tensor([dataset_id] * num_samples).to(self._device)
        dataset_names = [dataset_name] * num_samples
        label_types = [label_type] * num_samples

        if label_type == 'single':
            condition_config = HybridConditionConfig(
                dataset_ids=dataset_ids,
                dataset_names=dataset_names,
                label_types=label_types,
                single_labels=labels,
                multi_labels=None,
                single_mask=torch.ones(num_samples, dtype=torch.bool),
                multi_mask=torch.zeros(num_samples, dtype=torch.bool)
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
            return condition_kwargs.to_dict()
        else:  # multi-label
            condition_config = HybridConditionConfig(
                dataset_ids=dataset_ids,
                dataset_names=dataset_names,
                label_types=label_types,
                single_labels=None,
                multi_labels=labels,
                single_mask=torch.zeros(num_samples, dtype=torch.bool),
                multi_mask=torch.ones(num_samples, dtype=torch.bool)
            )
            condition_kwargs = ConditionConfig(
                condition_config=condition_config,
                use_hybrid_conditioning=self.use_hybrid_conditioning
            )
            return condition_kwargs.to_dict()

    def generate_dataset_specific_samples(self, dataset_id: int, dataset_name: str, num_samples_per_class: int = 4):
        """Generate samples for all classes of a specific dataset"""
        if not self.use_hybrid_conditioning:
            raise ValueError("This method is only applicable for Conditional VAE models with Hybrid Conditioning.")

        self._model.eval()
        generated_samples = []
        label_info = self._model.label_type_info.get(dataset_name, {})
        with torch.no_grad():
            if label_info.get(TYPE_NAME) == DatasetLabelType.SINGLE:
                for label_id in range(label_info.get(N_CLASS_NAME, 0)):
                    samples = self.predict(
                        num_samples=num_samples_per_class,
                        dataset_id=dataset_id,
                        label_type=DatasetLabelType.SINGLE.value,
                        labels=label_id,
                        dataset_name=dataset_name
                    )
                    generated_samples.append(samples)
            elif label_info.get(TYPE_NAME) == DatasetLabelType.MULTI:
                n_classes = label_info.get(N_CLASS_NAME, 0)
                # Generate random multi-label vectors
                for label_id in range(min(n_classes, 3)):
                    labels = torch.zeros(num_samples_per_class, n_classes).to(self._device)
                    labels[:, label_id] = 1.0  # Set the current label to 1
                    samples = self.predict(
                        num_samples=num_samples_per_class,
                        dataset_id=dataset_id,
                        label_type=DatasetLabelType.MULTI.value,
                        labels=labels,
                        dataset_name=dataset_name
                    )
                    generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)

    def load_parameters(self, path):
        """Load model parameters from a given path."""
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.to(self._device)

    def save_parameters(self, path):
        """Save model parameters to a given path."""
        torch.save(self._model.state_dict(), path)

    def reconstruct(self, x):
        """Reconstruct input images using the VAE model."""
        self._model.eval()
        x = x.to(self._device)
        with torch.no_grad():
            x_hat, mu, logvar = self._model(x)
            return x_hat, mu, logvar

    def save_checkpoint(self, path: PathLike, args: Dict[str, Any] = None):
        """Save the model and its training records to a checkpoint file."""
        from core.utils.training_records import save_checkpoint

        current_metrics = self.get_current_metrics_summary()

        self.records_manager = save_checkpoint(
            model=self._model,
            save_dir=path,
            args=args,
            metrics=current_metrics,
            records_manager=self.records_manager
        )

        return self.records_manager

    def load_checkpoint(self, checkpoint_path: PathLike, current_args, force_continue: bool = False):
        """Load the model and its training records from a checkpoint file."""
        from core.utils.training_records import load_checkpoint

        self.records_manager = load_checkpoint(
            model=self._model,
            load_dir=checkpoint_path,
            current_args=current_args,
            force_continue=force_continue
        )

        self._model.to(self._device)

        return self.records_manager

    def get_current_metrics_summary(self):
        """Get summary of current training metrics for final report."""
        train_losses = self._metrics['train']['loss']
        val_losses = self._metrics['validation']['loss']

        summary = {
            'epochs_completed': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_val_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else None
        }

        return summary

    @property
    def is_conditional_training(self) -> bool:
        """Check if the model is a Conditional VAE."""
        return self._model.model_type == VAEModelType.CVAE

    @property
    def use_hybrid_conditioning(self) -> bool:
        """Check if the model uses hybrid conditioning."""
        return self._model.use_hybrid_conditioning if hasattr(self._model, 'use_hybrid_conditioning') else False


def init_and_load_model(img_shape, latent_dim, checkpoint_path=None, device="cpu", args=None,
                        conditioning_info:dict = None):

    network = init_model_backbone(args, conditioning_info, img_shape, latent_dim)

    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logger.info(f"Initialized model: {network.__class__.__name__} with {total_params} trainable parameters")

    agent = VariationalAutoEncoder(model=network, device=device)

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            try:
                records_manager = agent.load_checkpoint(
                    checkpoint_path=args.checkpoint_path,
                    current_args=args,
                    force_continue=getattr(args, 'force_continue', False)
                )

                # Get info about previous training
                if records_manager.records:
                    latest_record = records_manager.get_latest_record()
                    logger.info(f"Loaded model from training session {latest_record.train_count}")
                    logger.info(f"Previous training timestamp: {latest_record.timestamp}")

                    if latest_record.metrics:
                        last_metrics = latest_record.metrics[-1]
                        logger.info(f"Previous best val loss: {last_metrics.get('best_val_loss', 'N/A')}")

            except ValueError as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return

            print(f"Model parameters loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist. Starting with a new model.")

    return agent


def init_model_backbone(args, conditioning_info, img_shape, latent_dim):
    ModelClass = get_model(args.model)

    model_kwargs = {
        'img_shape': img_shape,
        'latent_dim': latent_dim,
        'model_type': args.model,
    }

    # Add hybrid conditioning parameters
    if args.model == VAEModelType.CVAE and conditioning_info:
        model_kwargs.update({
            'num_datasets': conditioning_info['num_datasets'],
            'label_type_info': conditioning_info['label_type_info'],
            'dataset_embed_dim': 16,
            'class_embed_dim': 16,
            'use_hybrid_conditioning': True,
            'condition_dim': args.condition_dim,
        })

        # Fallback for traditional conditioning
        total_classes = sum(info['n_classes'] for info in conditioning_info['datasets_info'].values())
        model_kwargs['num_classes'] = total_classes
    return ModelClass(**model_kwargs)