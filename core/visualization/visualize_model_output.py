import logging
from os import PathLike
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import save_image, make_grid
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.utils.general import hyphenate_and_wrap_text

logger = logging.getLogger(__name__)


def generate_cvae_report(agent, artifacts_dir: PathLike = "cvae_report",
                         dataset_info: Optional[Dict[str, List]] = None, data_loader=None):
    """
    Generate images for the CVAE mode report

    """

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generate_class_conditioned_samples_grid(
        agent=agent,
        dataset_info=dataset_info,
        samples_per_class=5,
        fig_size=(12, 8),
        artifacts_dir=artifacts_dir,
    )

    generate_class_conditioned_samples_with_same_latent_grid(
        agent=agent,
        dataset_info=dataset_info,
        n_latent_vectors=5,
        fig_size=(12, 8),
        artifacts_dir=artifacts_dir,
    )

    generate_reconstruction_comparison(
        agent=agent,
        test_dataloader=data_loader,
        dataset_info=dataset_info,
        artifacts_dir=artifacts_dir,
        n_samples=8,
        fig_size=(12, 4),
    )

    logger.info("Completed generating CVAE report.")


def generate_class_conditioned_samples_grid(
        agent, dataset_info, samples_per_class: int = 5, fig_size: Tuple[int, int] = (12, 8),
        artifacts_dir: PathLike = "outputs"
):
    """
    Generate a grid of conditional samples showing different classes.
    Each row represents one class, each column shows different samples of that class.
    """
    if not agent.is_conditional_training:
        logger.warning("Agent is not conditional. Skipping conditional generation.")
        return

    class_labels = dataset_info.get('label', {})
    n_classes = len(class_labels)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    artifacts_dir = Path(artifacts_dir)

    # Generate samples for each class
    all_samples = []
    class_names = []

    for class_idx in range(n_classes):
        samples = agent.predict(num_samples=samples_per_class, labels=class_idx)
        all_samples.append(samples)
        class_names.append(class_labels.get(str(class_idx), f'Class {class_idx}'))

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)

    grid = make_grid(all_samples, nrow=samples_per_class, normalize=True, padding=2, pad_value=1.0)

    # Save the grid image
    save_path = artifacts_dir / "cvae_conditional_samples"
    save_image(grid, save_path.with_suffix(".png"))
    save_image(grid, save_path.with_suffix(".pdf"))

    # Create annotated version with class labels
    fig, ax = plt.subplots(figsize=fig_size)

    # Convert tensor to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[2] == 1:  # Grayscale
        grid_np = grid_np.squeeze(2)
        ax.imshow(grid_np, cmap='gray')
    else:
        ax.imshow(grid_np)

    # Add class labels on the left side
    img_height = grid_np.shape[0]
    row_height = img_height // n_classes

    # Calculate appropriate font size and wrap width based on number of classes
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for i, class_name in enumerate(class_names):
        y_pos = (i + 0.5) * row_height

        # Join lines with newline characters
        display_name = wrap_class_name(class_name, wrap_width=wrap_width)

        ax.text(-40, y_pos, display_name, rotation=0, ha='right', va='center',
                fontsize=label_fontsize, fontweight='bold', color='black',
                linespacing=1.2)

    ax.set_title(f'CVAE Conditional Generation - Samples by Class on {dataset_name}', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Adjust subplot parameters to make room for multi-line labels
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    save_path = artifacts_dir / "cvae_conditional_samples_labeled"
    plt.savefig(save_path.with_suffix(".png"), bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()

    logger.info(f"Conditional samples saved to {artifacts_dir}")


def generate_reconstruction_comparison(
        agent,
        test_dataloader,
        dataset_info: Optional[Dict[str, List]] = None,
        artifacts_dir: PathLike = "outputs",
        n_samples: int = 8,
        fig_size: Tuple[int, int] = (12, 8),
):
    """
    Generate reconstruction comparison using real test data images.
    Shows original images vs their reconstructions side by side.
    """
    agent._model.eval()
    device = agent._device
    artifacts_dir = Path(artifacts_dir)

    # Get a batch of test images
    test_batch = next(iter(test_dataloader))
    if agent.is_conditional_training:
        test_images, test_labels = test_batch
        test_images = test_images[:n_samples].to(device)
        test_labels = test_labels[:n_samples].to(device).squeeze(dim=1)
    else:
        test_images, _ = test_batch
        test_images = test_images[:n_samples].to(device)
        test_labels = None

    # Generate reconstructions
    with torch.no_grad():
        if agent.is_conditional_training:
            reconstructed, mu, logvar = agent._model(test_images, test_labels)
        else:
            reconstructed, mu, logvar = agent._model(test_images)

    # Create comparison visualization
    fig, axes = plt.subplots(2, n_samples, figsize=fig_size)

    # Get class labels for annotation (if available)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    class_labels = dataset_info.get('label', {}) if agent.is_conditional_training else None
    n_classes = len(class_labels)
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for i in range(n_samples):
        # Original image (top row)
        orig_img = test_images[i].cpu().permute(1, 2, 0).numpy()
        if orig_img.shape[2] == 1:  # Grayscale
            orig_img = orig_img.squeeze(2)
            axes[0, i].imshow(orig_img, cmap='gray')
        else:
            axes[0, i].imshow(orig_img)

        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].text(-0.1, 0.5, 'Original', transform=axes[0, i].transAxes,
                            rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')

        # Add class label if available
        if class_labels and test_labels is not None:
            class_idx = test_labels[i].item()
            class_name = class_labels.get(str(class_idx), f'Class {class_idx}')
            wrapped_name = wrap_class_name(class_name, wrap_width=wrap_width)
            axes[0, i].set_title(wrapped_name, fontsize=label_fontsize, fontweight='bold')

        # Reconstructed image (bottom row)
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        if recon_img.shape[2] == 1:  # Grayscale
            recon_img = recon_img.squeeze(2)
            axes[1, i].imshow(recon_img, cmap='gray')
        else:
            axes[1, i].imshow(recon_img)

        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].text(-0.1, 0.5, 'Reconstructed', transform=axes[1, i].transAxes,
                            rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')

    # Calculate reconstruction loss for annotation
    mse_loss = torch.nn.functional.mse_loss(reconstructed, test_images, reduction='mean')

    model_type = "CVAE" if agent.is_conditional_training else "VAE"
    plt.suptitle(f'{model_type} Reconstruction Comparison on {dataset_name}\n'
                 f'Original vs Reconstructed Images (MSE Loss: {mse_loss:.4f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.08)
    save_path = artifacts_dir / f"{model_type.lower()}_reconstruction_comparison"
    plt.savefig(save_path.with_suffix(".png"), bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()


def generate_class_conditioned_samples_with_same_latent_grid(
        agent,
        dataset_info,
        n_latent_vectors: int = 5,
        fig_size: Tuple[int, int] = (12, 8),
        artifacts_dir: PathLike = "outputs"
):
    """
    Generate comparison showing same latent vectors with different conditional labels.
    """
    if not agent.is_conditional_training:
        logger.warning("Agent is not conditional. Skipping conditional reconstruction comparison.")
        return

    class_labels = dataset_info.get('label', {})
    n_classes = len(class_labels)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    latent_dim = agent._model.latent_dim
    device = agent._device

    artifacts_dir = Path(artifacts_dir)

    # Generate fixed latent vectors
    with torch.no_grad():
        fixed_latent_vectors = torch.randn(n_latent_vectors, latent_dim).to(device)

        # Generate samples for each latent vector with each class condition
        all_samples = []

        for i, z in enumerate(fixed_latent_vectors):
            row_samples = []
            for class_idx in range(n_classes):
                # Use same latent vector but different class condition
                z_expanded = z.unsqueeze(0)  # Add batch dimension
                class_tensor = torch.tensor([class_idx]).to(device)
                sample = agent._model.decode(z_expanded, class_tensor)
                row_samples.append(sample)

            # Concatenate samples for this latent vector
            row_samples = torch.cat(row_samples, dim=0)
            all_samples.append(row_samples)

    # Create visualization
    fig, axes = plt.subplots(n_latent_vectors, n_classes, figsize=fig_size)

    # Handle single row case
    if n_latent_vectors == 1:
        axes = axes.reshape(1, -1)

    # Get class names for headers
    class_names = [class_labels.get(str(i), f'Class {i}') for i in range(n_classes)]

    # Calculate appropriate font size and wrap width based on number of classes
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for row_idx, row_samples in enumerate(all_samples):
        for col_idx, sample in enumerate(row_samples):
            ax = axes[row_idx, col_idx]

            # Display sample
            sample_img = sample.cpu().permute(1, 2, 0).numpy()
            if sample_img.shape[2] == 1:  # Grayscale
                sample_img = sample_img.squeeze(2)
                ax.imshow(sample_img, cmap='gray')
            else:
                ax.imshow(sample_img)

            ax.axis('off')

            # Add class name as column header (only for first row)
            if row_idx == 0:
                # Wrap long class names into multiple lines
                wrapped_name = wrap_class_name(class_names[col_idx], wrap_width=wrap_width)

                ax.set_title(wrapped_name, fontsize=label_fontsize, fontweight='bold', pad=10)

            # Add latent vector label on the left (only for first column)
            if col_idx == 0:
                ax.text(-0.35, 0.5, f'Latent\nVector {row_idx + 1}',
                        transform=ax.transAxes, rotation=0, ha='center', va='center',
                        fontsize=9, fontweight='bold')

    plt.suptitle(f'CVAE Conditional Reconstruction Comparison on {dataset_name}\n'
                 'Same Latent Vectors with Different Class Conditions',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.1)

    save_path = artifacts_dir / "cvae_conditional_reconstruction_comparison"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()

    # Also create a detailed version with annotations TODO
    # create_detailed_reconstruction_comparison(all_samples, class_names, artifacts_dir, n_latent_vectors)

    logger.info(f"Conditional reconstruction comparison saved to {artifacts_dir}")


def compute_wrap_width_and_label_font_size(n_classes: int) -> tuple[int, int]:
    """
    Compute proper label font size and wrap width based on number of classes.
    """
    if n_classes <= 5:
        label_fontsize = 10
        wrap_width = 12
    elif n_classes <= 10:
        label_fontsize = 9
        wrap_width = 10
    else:
        label_fontsize = 8
        wrap_width = 8
    return label_fontsize, wrap_width


def wrap_class_name(class_name: str, wrap_width: int = 10) -> str:
    """
    Wrap class name to fit within a specified width.
    :param class_name: The class name to wrap.
    :param wrap_width: The maximum width of each line.
    :return: Wrapped class name.
    """
    wrapped_lines = hyphenate_and_wrap_text(class_name, wrap_width=wrap_width)
    # If still too many lines, truncate and add ellipsis
    if len(wrapped_lines) > 3:
        wrapped_lines = wrapped_lines[:2] + [wrapped_lines[2][:wrap_width - 3] + "..."]

    return '\n'.join(wrapped_lines)
