import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def show_technical_details():
    """
    Display technical details about CNN-JEPA implementation
    """
    st.header("Technical Details of CNN-JEPA")
    
    # Model Architecture
    st.subheader("Model Architecture")
    
    st.markdown("""
    CNN-JEPA consists of three main components:
    
    1. **Context Encoder (Sparse CNN)**
       - Architecture: Modified ResNet-50
       - Input: Masked image
       - Output: Context features with mask information
       - Key Modifications:
         - Propagates mask through the network
         - Applies convolutions only to visible regions
         - Downsamples mask along with feature maps
    
    2. **Predictor (Depthwise Separable CNN)**
       - Architecture: Fully convolutional network with depthwise separable convolutions
       - Input: Context features
       - Output: Predicted target features
       - Design:
         - 4 depthwise separable convolutional blocks
         - Maintains spatial dimensions of feature maps
         - Parameter-efficient (fewer parameters than standard convolutions)
    
    3. **Target Encoder (Standard CNN)**
       - Architecture: Standard ResNet-50
       - Input: Unmasked image
       - Output: Target features
       - Update: EMA of context encoder (typically with decay 0.996-0.999)
    """)
    
    # Sparse CNN Implementation
    st.subheader("Sparse CNN Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sparse Convolution Block**
        
        ```python
        class SparseCNNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, 
                        kernel_size=3, stride=1, padding=1):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size, stride, padding)
                self.bn = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
            def forward(self, x, mask=None):
                x = self.conv(x)
                
                if mask is not None:
                    # Apply mask to output
                    x = x * mask
                    
                    # Downsample mask if stride > 1
                    if self.conv.stride[0] > 1:
                        mask = F.max_pool2d(
                            mask, 
                            kernel_size=self.conv.stride[0],
                            stride=self.conv.stride[0]
                        )
                
                x = self.bn(x)
                x = self.relu(x)
                
                return x, mask
        ```
        """)
    
    with col2:
        st.markdown("""
        **Sparse Residual Block**
        
        ```python
        class SparseResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, 
                        stride=1, downsample=None):
                super().__init__()
                self.conv1 = SparseCNNBlock(
                    in_channels, out_channels, 
                    stride=stride
                )
                self.conv2 = nn.Conv2d(
                    out_channels, out_channels, 
                    kernel_size=3, padding=1
                )
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                
            def forward(self, x, mask=None):
                identity = x
                
                out, mask_out = self.conv1(x, mask)
                
                out = self.conv2(out)
                if mask_out is not None:
                    out = out * mask_out
                    
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                    if mask is not None and self.stride > 1:
                        mask_identity = F.max_pool2d(
                            mask, 
                            kernel_size=self.stride,
                            stride=self.stride
                        )
                        identity = identity * mask_identity
                
                out += identity
                out = self.relu(out)
                
                return out, mask_out
        ```
        """)
    
    # Depthwise Separable Convolution Implementation
    st.subheader("Depthwise Separable Convolution Implementation")
    
    st.markdown("""
    ```python
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            # Depthwise convolution (one filter per input channel)
            self.depthwise = nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                groups=in_channels  # Key parameter for depthwise conv
            )
            
            # Pointwise convolution (1x1 conv to mix channels)
            self.pointwise = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1
            )
            
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            x = self.relu(x)
            return x
    ```
    
    **Predictor Implementation**
    
    ```python
    class CNNJEPAPredictor(nn.Module):
        def __init__(self, in_channels=2048, hidden_channels=1024, 
                    out_channels=2048, num_blocks=4):
            super().__init__()
            
            layers = []
            # First layer: in_channels -> hidden_channels
            layers.append(DepthwiseSeparableConv(in_channels, hidden_channels))
            
            # Middle layers: hidden_channels -> hidden_channels
            for _ in range(num_blocks - 2):
                layers.append(DepthwiseSeparableConv(hidden_channels, hidden_channels))
            
            # Last layer: hidden_channels -> out_channels
            layers.append(DepthwiseSeparableConv(hidden_channels, out_channels))
            
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x)
    ```
    """)
    
    # Masking Strategy
    st.subheader("Masking Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **CNN-JEPA Masking**
        
        CNN-JEPA uses a simplified masking strategy compared to I-JEPA:
        
        - **Masking Ratio**: Typically ~40% of the image is masked
        - **Block Size**: Usually 32×32 pixels
        - **Target Blocks**: Typically a single target block (unlike I-JEPA's multiple targets)
        - **Mask Propagation**: The mask is propagated through the network, being downsampled
          along with the feature maps
        
        This strategy is designed to work well with the downsampling nature of CNNs and to
        provide a clear prediction target.
        """)
        
        # Create a simple masking visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        mask = np.ones((14, 14))  # 14x14 grid (224/16 patches)
        
        # Create context region (visible)
        mask[3:7, 3:7] = 0  # Single target block
        
        # Plot mask
        cmap = plt.cm.get_cmap('viridis', 2)
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Masked (Target)', 'Visible (Context)'])
        
        ax.set_title('CNN-JEPA Masking Strategy')
        ax.axis('off')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **Mask Implementation**
        
        ```python
        def create_cnn_mask(height, width, masking_ratio=0.4, 
                          block_size=32, num_target_blocks=1):
            # Calculate grid dimensions
            h_blocks = height // block_size
            w_blocks = width // block_size
            total_blocks = h_blocks * w_blocks
            
            # Initialize masks
            context_mask = np.ones((height, width))
            target_mask = np.zeros((height, width))
            
            # Calculate number of blocks to mask
            num_blocks_to_mask = int(total_blocks * masking_ratio)
            
            # Create a grid of block indices
            block_indices = np.arange(total_blocks)
            np.random.shuffle(block_indices)
            
            # Select target blocks
            target_block_indices = block_indices[:num_target_blocks]
            
            # Select remaining blocks to mask
            other_masked_block_indices = block_indices[
                num_target_blocks:num_blocks_to_mask
            ]
            
            # Apply target blocks to masks
            for idx in target_block_indices:
                h_idx = idx // w_blocks
                w_idx = idx % w_blocks
                
                h_start = h_idx * block_size
                h_end = min((h_idx + 1) * block_size, height)
                w_start = w_idx * block_size
                w_end = min((w_idx + 1) * block_size, width)
                
                context_mask[h_start:h_end, w_start:w_end] = 0
                target_mask[h_start:h_end, w_start:w_end] = 1
            
            # Apply other masked blocks to context mask
            for idx in other_masked_block_indices:
                h_idx = idx // w_blocks
                w_idx = idx % w_blocks
                
                h_start = h_idx * block_size
                h_end = min((h_idx + 1) * block_size, height)
                w_start = w_idx * block_size
                w_end = min((w_idx + 1) * block_size, width)
                
                context_mask[h_start:h_end, w_start:w_end] = 0
            
            return context_mask, target_mask
        ```
        """)
    
    # Loss Function
    st.subheader("Loss Function")
    
    st.markdown("""
    CNN-JEPA uses a simple L2 loss between predicted and target features, but only at masked locations:
    
    $$L = \\frac{1}{N_m} \\sum_{i \\in \\mathcal{M}} \\| f_\\theta(x_i^{ctx}) - g_\\xi(x_i) \\|_2^2$$
    
    Where:
    - $f_\\theta$ is the predictor applied to context features
    - $g_\\xi$ is the target encoder
    - $x_i^{ctx}$ is the masked input with only context regions visible
    - $x_i$ is the original unmasked input
    - $\\mathcal{M}$ is the set of masked locations
    - $N_m$ is the number of masked locations
    
    ```python
    def compute_loss(pred_features, target_features, final_mask):
        # Invert mask (1 = masked, 0 = visible)
        mask = 1 - final_mask
        
        # Calculate L2 distance only at masked locations
        loss = torch.sum((pred_features - target_features)**2 * mask) / (torch.sum(mask) + 1e-6)
        
        return loss
    ```
    
    This loss encourages the model to predict the target features from the context features only
    at the masked locations, focusing the learning on the prediction task.
    """)
    
    # Training Details
    st.subheader("Training Details")
    
    st.markdown("""
    **Pretraining Dataset**
    
    CNN-JEPA is typically pretrained on ImageNet-1K:
    - 1.28 million training images
    - 1000 classes
    
    **Optimization**
    
    - Optimizer: AdamW
    - Learning rate: 1.5e-4 with cosine decay
    - Weight decay: 0.05
    - Batch size: 2048
    - Training epochs: 300-800
    - Hardware: 8-16 GPUs
    
    **Data Augmentation**
    
    Unlike many other self-supervised methods, CNN-JEPA requires only minimal data augmentations:
    - Random resized crop
    - No color jitter
    - No flips
    - No rotations
    
    This simplicity is one of the advantages of CNN-JEPA compared to other methods.
    """)
    
    # Evaluation Protocol
    st.subheader("Evaluation Protocol")
    
    st.markdown("""
    CNN-JEPA is evaluated using several protocols:
    
    1. **Linear Probing**
       - Freeze the pretrained encoder
       - Train a linear classifier on top
       - Evaluate on ImageNet validation set
    
    2. **k-NN Classification**
       - Extract features from the pretrained encoder
       - Classify test images using k-nearest neighbors
       - No additional training required
    
    3. **Fine-Tuning**
       - Initialize with pretrained weights
       - Fine-tune the entire model
       - Evaluate on ImageNet validation set
    
    4. **Transfer Learning**
       - Initialize with pretrained weights
       - Fine-tune on downstream tasks
    """)
    
    # Results
    st.subheader("Results")
    
    # Create a simple bar chart comparing methods
    methods = ['CNN-JEPA', 'I-JEPA', 'BYOL', 'SimCLR', 'VICReg']
    linear_probe = [73.3, 71.5, 72.8, 71.2, 72.1]  # Example values for ImageNet-100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, linear_probe, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom')
    
    ax.set_ylabel('ImageNet-100 Linear Probe Accuracy (%)')
    ax.set_title('Comparison of Self-Supervised Methods (ResNet-50)')
    
    st.pyplot(fig)
    
    st.markdown("""
    **Key Results**
    
    - CNN-JEPA outperforms I-JEPA with ViT backbones on ImageNet-100 using a ResNet-50
    - It achieves competitive results with established CNN-based SSL methods like BYOL, SimCLR, and VICReg
    - CNN-JEPA requires 17-35% less training time compared to other methods for the same number of epochs
    - The method achieves these results with minimal data augmentations and without a separate projector network
    """)
    
    # Limitations and Future Work
    st.subheader("Limitations and Future Work")
    
    st.markdown("""
    **Current Limitations**
    
    - Implementation of sparse convolutions can be inefficient (masking standard convolution outputs)
    - The approach may not fully leverage the inductive biases of CNNs
    - Performance on very large datasets and models has not been fully explored
    
    **Future Directions**
    
    - More efficient implementations of sparse convolutions
    - Exploring different masking strategies specifically designed for CNNs
    - Combining CNN-JEPA with other self-supervised approaches
    - Applying CNN-JEPA to specialized domains (medical imaging, satellite imagery, etc.)
    - Scaling to larger models and datasets
    """)
    
    # Conclusion
    st.subheader("Conclusion")
    
    st.markdown("""
    CNN-JEPA represents a significant advancement in self-supervised learning for CNNs:
    
    - It successfully adapts the JEPA framework to CNNs, addressing the inherent challenges of
      using masked inputs and predictive architectures with convolutional networks.
    
    - The sparse CNN encoder and efficient predictor design are key innovations that make the
      approach work effectively with CNNs.
    
    - The method achieves strong performance on standard benchmarks while requiring less training
      time and a simpler design compared to other CNN-based SSL methods.
    
    This demonstration provides an interactive way to explore the key concepts of CNN-JEPA and
    understand how it processes image data.
    """)
