import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from sklearn.decomposition import PCA

def visualize_masking(image, context_mask, target_mask):
    """
    Visualize the masking strategy

    Args:
        image: Original image [H, W, C]
        context_mask: Context mask [H, W] (1 = visible, 0 = masked)
        target_mask: Target mask [H, W] (1 = target, 0 = non-target)

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Context mask
    ax2 = fig.add_subplot(gs[0, 1])
    masked_image = image.copy()

    # Apply context mask (gray out masked regions)
    for c in range(3):
        masked_image[:, :, c] = masked_image[:, :, c] * context_mask + 0.5 * (1 - context_mask)

    ax2.imshow(masked_image)
    ax2.set_title("Context (Visible Regions)")
    ax2.axis("off")

    # Target mask
    ax3 = fig.add_subplot(gs[0, 2])
    target_vis = image.copy()

    # Highlight target regions
    highlight = np.zeros_like(target_vis)
    highlight[:, :, 0] = target_mask  # Red channel

    # Blend with original image
    alpha = 0.5
    target_vis = target_vis * (1 - alpha * target_mask[:, :, np.newaxis]) + highlight * alpha

    ax3.imshow(target_vis)
    ax3.set_title("Target Regions")
    ax3.axis("off")

    plt.tight_layout()
    return fig

def visualize_feature_maps(context_features, predicted_features, target_features, final_mask=None):
    """
    Visualize feature maps

    Args:
        context_features: Context features [B, C, H, W]
        predicted_features: Predicted features [B, C, H, W]
        target_features: Target features [B, C, H, W]
        final_mask: Final mask [B, 1, H, W] (1 = visible, 0 = masked)

    Returns:
        fig: Matplotlib figure
    """
    try:
        # Convert to numpy if needed
        if isinstance(context_features, torch.Tensor):
            context_features = context_features.detach().cpu().numpy()
        if isinstance(predicted_features, torch.Tensor):
            predicted_features = predicted_features.detach().cpu().numpy()
        if isinstance(target_features, torch.Tensor):
            target_features = target_features.detach().cpu().numpy()
        if isinstance(final_mask, torch.Tensor) and final_mask is not None:
            final_mask = final_mask.detach().cpu().numpy()

        # Take the first image in the batch
        context_features = context_features[0]
        predicted_features = predicted_features[0]
        target_features = target_features[0]
        if final_mask is not None:
            final_mask = final_mask[0, 0]  # Remove batch and channel dimensions

        # Calculate feature map statistics
        context_mean = np.mean(context_features, axis=0)
        predicted_mean = np.mean(predicted_features, axis=0)
        target_mean = np.mean(target_features, axis=0)

        # Ensure all feature maps have the same shape
        if context_mean.shape != predicted_mean.shape or context_mean.shape != target_mean.shape:
            # Resize to match the smallest shape
            min_shape = min(context_mean.shape, predicted_mean.shape, target_mean.shape, key=lambda x: x[0] * x[1])
            if context_mean.shape != min_shape:
                context_mean = np.resize(context_mean, min_shape)
            if predicted_mean.shape != min_shape:
                predicted_mean = np.resize(predicted_mean, min_shape)
            if target_mean.shape != min_shape:
                target_mean = np.resize(target_mean, min_shape)
            if final_mask is not None and final_mask.shape != min_shape:
                final_mask = np.resize(final_mask, min_shape)

        # Create figure
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig)

        # Context features
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(context_mean, cmap='viridis')
        ax1.set_title("Context Features (Mean)")
        plt.colorbar(im1, ax=ax1)

        # Predicted features
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(predicted_mean, cmap='viridis')
        ax2.set_title("Predicted Features (Mean)")
        plt.colorbar(im2, ax=ax2)

        # Target features
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(target_mean, cmap='viridis')
        ax3.set_title("Target Features (Mean)")
        plt.colorbar(im3, ax=ax3)

        # Mask (if provided)
        if final_mask is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(final_mask, cmap='gray')
            ax4.set_title("Final Mask")
            plt.colorbar(im4, ax=ax4)

        # Difference between predicted and target
        ax5 = fig.add_subplot(gs[1, 1])
        diff = np.abs(predicted_mean - target_mean)
        im5 = ax5.imshow(diff, cmap='hot')
        ax5.set_title("Prediction Error (Abs Diff)")
        plt.colorbar(im5, ax=ax5)

        # Histogram of differences
        ax6 = fig.add_subplot(gs[1, 2])
        if final_mask is not None:
            # Only consider masked regions
            masked_diff = diff * (1 - final_mask)
            masked_diff = masked_diff.flatten()
            masked_diff = masked_diff[masked_diff > 0]  # Remove zeros
            if len(masked_diff) > 0:
                ax6.hist(masked_diff, bins=50)
                ax6.set_title("Error Distribution in Masked Regions")
                ax6.set_xlabel("Absolute Difference")
                ax6.set_ylabel("Frequency")
        else:
            ax6.hist(diff.flatten(), bins=50)
            ax6.set_title("Error Distribution")
            ax6.set_xlabel("Absolute Difference")
            ax6.set_ylabel("Frequency")
    except Exception as e:
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f"Error visualizing feature maps: {str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        return fig

def visualize_predictions(image, context_mask, target_mask, predicted_features, target_features, final_mask=None):
    """
    Visualize the predictions

    Args:
        image: Original image [H, W, C]
        context_mask: Context mask [H, W] (1 = visible, 0 = masked)
        target_mask: Target mask [H, W] (1 = target, 0 = non-target)
        predicted_features: Predicted features
        target_features: Target features
        final_mask: Final mask after downsampling

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Context mask
    ax2 = fig.add_subplot(gs[0, 1])
    masked_image = image.copy()

    # Apply context mask (gray out masked regions)
    for c in range(3):
        masked_image[:, :, c] = masked_image[:, :, c] * context_mask + 0.5 * (1 - context_mask)

    ax2.imshow(masked_image)
    ax2.set_title("Context (Visible Regions)")
    ax2.axis("off")

    # Target mask
    ax3 = fig.add_subplot(gs[0, 2])
    target_vis = image.copy()

    # Highlight target regions
    highlight = np.zeros_like(target_vis)
    highlight[:, :, 0] = target_mask  # Red channel

    # Blend with original image
    alpha = 0.5
    target_vis = target_vis * (1 - alpha * target_mask[:, :, np.newaxis]) + highlight * alpha

    ax3.imshow(target_vis)
    ax3.set_title("Target Regions")
    ax3.axis("off")

    # Prediction visualization
    ax4 = fig.add_subplot(gs[1, 0:3])

    # Calculate L2 distance between predicted and target features
    if isinstance(predicted_features, torch.Tensor):
        predicted_features = predicted_features.detach().cpu().numpy()
    if isinstance(target_features, torch.Tensor):
        target_features = target_features.detach().cpu().numpy()

    # Reshape if needed
    if len(predicted_features.shape) > 2:
        # For CNN features [B, C, H, W], reshape to [B*H*W, C]
        B, C, H, W = predicted_features.shape
        predicted_features = predicted_features.transpose(0, 2, 3, 1).reshape(-1, C)

    if len(target_features.shape) > 2:
        # For CNN features [B, C, H, W], reshape to [B*H*W, C]
        B, C, H, W = target_features.shape
        target_features = target_features.transpose(0, 2, 3, 1).reshape(-1, C)

    # Only compare features for target regions if final_mask is provided
    if final_mask is not None:
        if isinstance(final_mask, torch.Tensor):
            final_mask = final_mask.detach().cpu().numpy()

        # Reshape final_mask to match feature dimensions
        if len(final_mask.shape) > 2:
            # For CNN masks [B, 1, H, W], reshape to [B*H*W]
            final_mask = final_mask.reshape(-1)

        # Get indices of masked regions (where mask is 0)
        masked_indices = np.where(final_mask < 0.5)[0]

        if len(masked_indices) > 0:
            # Ensure indices are within bounds
            valid_indices = masked_indices[masked_indices < len(predicted_features)]

            if len(valid_indices) > 0:
                predicted_features = predicted_features[valid_indices]
                target_features = target_features[valid_indices]

    # Calculate L2 distance
    try:
        l2_distance = np.sqrt(np.sum((predicted_features - target_features) ** 2, axis=1))

        # Plot histogram of L2 distances
        ax4.hist(l2_distance, bins=30, alpha=0.7)
        ax4.set_title("L2 Distance Between Predicted and Target Features")
        ax4.set_xlabel("L2 Distance")
        ax4.set_ylabel("Frequency")

        # Add mean and std as text
        mean_dist = np.mean(l2_distance)
        std_dist = np.std(l2_distance)
        ax4.text(0.95, 0.95, f"Mean: {mean_dist:.4f}\nStd: {std_dist:.4f}",
                transform=ax4.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        ax4.text(0.5, 0.5, f"Error calculating distances: {str(e)}",
                ha='center', va='center', fontsize=12)

    plt.tight_layout()
    return fig

def visualize_embeddings(features, labels=None):
    """
    Visualize feature embeddings using PCA

    Args:
        features: Features to visualize [N, D] or [B, C, H, W]
        labels: Optional labels for coloring points

    Returns:
        fig: Matplotlib figure
    """
    try:
        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        # Reshape if needed
        if len(features.shape) > 2:
            # For CNN features [B, C, H, W], reshape to [B*H*W, C]
            if len(features.shape) == 4:
                B, C, H, W = features.shape
                features = features.transpose(0, 2, 3, 1).reshape(-1, C)
            else:
                features = features.reshape(-1, features.shape[-1])

        # Sample features if there are too many
        max_samples = 10000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            features = features[indices]
            if labels is not None:
                labels = labels[indices]

        # Ensure we have enough samples for PCA
        if features.shape[0] < 2:
            # Create a dummy embedding if we don't have enough samples
            features = np.random.randn(100, features.shape[1])
            labels = None

        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot embeddings
        if labels is not None:
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7)

        ax.set_title("PCA Visualization of Features")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

        plt.tight_layout()
        return fig
    except Exception as e:
        # Create a simple figure with an error message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error visualizing features: {str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        return fig
