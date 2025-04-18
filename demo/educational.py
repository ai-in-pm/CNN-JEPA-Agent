import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def create_architecture_diagram():
    """
    Create a simple architecture diagram for CNN-JEPA
    
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Define components
    components = {
        'image': (0.05, 0.5, 0.15, 0.3),
        'masked_image': (0.05, 0.1, 0.15, 0.3),
        'context_encoder': (0.3, 0.1, 0.15, 0.3),
        'target_encoder': (0.3, 0.5, 0.15, 0.3),
        'predictor': (0.55, 0.1, 0.15, 0.3),
        'target_features': (0.55, 0.5, 0.15, 0.3),
        'loss': (0.8, 0.3, 0.1, 0.1)
    }
    
    # Draw components
    for name, (x, y, w, h) in components.items():
        if name == 'loss':
            circle = plt.Circle((x + w/2, y + h/2), 0.05, fill=True, color='red', alpha=0.7)
            plt.gca().add_patch(circle)
            plt.text(x + w/2, y + h/2, 'L2', ha='center', va='center', fontsize=10, color='white')
        elif 'encoder' in name:
            rect = plt.Rectangle((x, y), w, h, fill=True, alpha=0.7, color='skyblue')
            plt.gca().add_patch(rect)
            plt.text(x + w/2, y + h/2, name.replace('_', '\n'), ha='center', va='center', fontsize=10)
            # Add CNN architecture details
            if name == 'context_encoder':
                plt.text(x + w/2, y - 0.05, 'Sparse CNN', ha='center', va='center', fontsize=8)
            else:
                plt.text(x + w/2, y - 0.05, 'Standard CNN', ha='center', va='center', fontsize=8)
        elif name == 'predictor':
            rect = plt.Rectangle((x, y), w, h, fill=True, alpha=0.7, color='lightgreen')
            plt.gca().add_patch(rect)
            plt.text(x + w/2, y + h/2, name.replace('_', '\n'), ha='center', va='center', fontsize=10)
            plt.text(x + w/2, y - 0.05, 'Depthwise Separable CNN', ha='center', va='center', fontsize=8)
        else:
            rect = plt.Rectangle((x, y), w, h, fill=True, alpha=0.7, color='lightgray')
            plt.gca().add_patch(rect)
            plt.text(x + w/2, y + h/2, name.replace('_', '\n'), ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        ('image', 'target_encoder'),
        ('image', 'masked_image'),
        ('masked_image', 'context_encoder'),
        ('context_encoder', 'predictor'),
        ('predictor', 'loss'),
        ('target_encoder', 'target_features'),
        ('target_features', 'loss')
    ]
    
    for start, end in arrows:
        start_x = components[start][0] + components[start][2]
        start_y = components[start][1] + components[start][3]/2
        
        end_x = components[end][0]
        end_y = components[end][1] + components[end][3]/2
        
        # Special case for the loss arrows
        if end == 'loss':
            end_x = components[end][0] + components[end][2]/2
            end_y = components[end][1] + components[end][3]/2
        
        plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                 head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
    
    # Add EMA update arrow
    ema_start_x = components['context_encoder'][0] + components['context_encoder'][2]/2
    ema_start_y = components['context_encoder'][1]
    ema_end_x = components['target_encoder'][0] + components['target_encoder'][2]/2
    ema_end_y = components['target_encoder'][1] + components['target_encoder'][3]
    
    plt.arrow(ema_start_x, ema_start_y, 0, ema_end_y - ema_start_y - 0.1, 
             head_width=0.02, head_length=0.02, fc='blue', ec='blue', length_includes_head=True, linestyle='dashed')
    plt.text(ema_start_x + 0.02, (ema_start_y + ema_end_y)/2, 'EMA\nUpdate', ha='left', va='center', fontsize=8, color='blue')
    
    # Set limits and remove axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    return fig

def create_sparse_cnn_diagram():
    """
    Create a diagram illustrating sparse CNN operation
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Input with mask
    input_img = np.ones((8, 8))
    mask = np.ones((8, 8))
    mask[2:6, 2:6] = 0  # Masked region
    
    masked_input = input_img.copy()
    masked_input[mask == 0] = 0.3  # Gray out masked regions
    
    axs[0].imshow(masked_input, cmap='gray')
    axs[0].set_title('Masked Input')
    axs[0].axis('off')
    
    # Sparse convolution
    sparse_output = np.ones((6, 6))
    sparse_mask = np.ones((6, 6))
    sparse_mask[1:5, 1:5] = 0  # Masked region after convolution
    sparse_output[sparse_mask == 0] = 0.3  # Gray out masked regions
    
    axs[1].imshow(sparse_output, cmap='gray')
    axs[1].set_title('Sparse Convolution Output')
    axs[1].axis('off')
    
    # Downsampled mask
    downsampled_mask = np.ones((4, 4))
    downsampled_mask[1:3, 1:3] = 0  # Masked region after downsampling
    
    axs[2].imshow(downsampled_mask, cmap='gray')
    axs[2].set_title('Downsampled Mask')
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig

def create_depthwise_separable_diagram():
    """
    Create a diagram illustrating depthwise separable convolution
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Standard convolution
    standard_input = np.ones((6, 6, 3))  # 3 channels
    standard_output = np.ones((4, 4, 4))  # 4 output channels
    
    # Create a simple visualization
    axs[0].imshow(np.sum(standard_input, axis=2)/3, cmap='Blues')
    axs[0].set_title('Input Feature Map\n(C channels)')
    axs[0].axis('off')
    
    # Depthwise convolution
    depthwise_output = np.ones((4, 4, 3))  # Same number of channels as input
    
    axs[1].imshow(np.sum(depthwise_output, axis=2)/3, cmap='Greens')
    axs[1].set_title('Depthwise Convolution\n(C channels)')
    axs[1].axis('off')
    
    # Pointwise convolution
    pointwise_output = np.ones((4, 4, 4))  # 4 output channels
    
    axs[2].imshow(np.sum(pointwise_output, axis=2)/4, cmap='Reds')
    axs[2].set_title('Pointwise Convolution\n(C\' channels)')
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig

def show_educational_content():
    """
    Display educational content about CNN-JEPA
    """
    st.header("Understanding CNN-JEPA")
    
    # Introduction
    st.subheader("Introduction")
    st.write("""
    CNN-JEPA (Convolutional Neural Network Joint-Embedding Predictive Architecture) is a self-supervised 
    learning method that adapts the JEPA framework, previously successful with Vision Transformers (ViTs), 
    to Convolutional Neural Networks (CNNs).
    
    The method addresses the inherent challenges of using masked inputs and predictive architectures with CNNs,
    introducing several key innovations to make the JEPA approach work effectively with convolutional architectures.
    """)
    
    # Key Innovations
    st.subheader("Key Innovations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sparse CNN Encoder**
        
        CNN-JEPA employs a sparse CNN architecture to effectively process masked image inputs:
        - Handles masked inputs by propagating the mask through the network
        - Produces masked feature maps that respect the original masking pattern
        - Adapts standard CNN operations to work with sparse inputs
        """)
        
        st.markdown("""
        **Efficient Predictor Design**
        
        Introduces a fully convolutional predictor using depthwise separable convolutions:
        - Parameter-efficient design for high-dimensional feature maps
        - Reduces computational cost compared to standard convolutions
        - Maintains spatial relationships in the feature maps
        """)
    
    with col2:
        st.markdown("""
        **CNN-Compatible Masking**
        
        Uses a simplified multi-block masking strategy adapted for CNN downsampling:
        - Typically predicts a single target region (unlike I-JEPA's multiple targets)
        - Accounts for the downsampling nature of CNN feature maps
        - Ensures mask alignment with network architecture
        """)
        
        st.markdown("""
        **Architectural Simplicity**
        
        CNN-JEPA achieves strong performance with a simpler design:
        - Requires only minimal data augmentations (random resized crop)
        - Does not need a separate projector network
        - Requires 17-35% less training time compared to other CNN SSL methods
        """)
    
    # Architecture Diagram
    st.subheader("CNN-JEPA Architecture")
    
    # Create a simple architecture diagram
    fig = create_architecture_diagram()
    st.pyplot(fig)
    
    # Sparse CNN Explanation
    st.subheader("Sparse CNN Encoder")
    st.write("""
    The sparse CNN encoder is a key innovation in CNN-JEPA. It allows the network to process masked inputs
    and produce feature maps that respect the original masking pattern.
    """)
    
    # Create a diagram illustrating sparse CNN operation
    fig = create_sparse_cnn_diagram()
    st.pyplot(fig)
    
    st.markdown("""
    **How Sparse CNNs Work:**
    
    1. **Mask Propagation**: The input mask is propagated through the network, ensuring that masked regions
       remain masked in the feature maps.
    
    2. **Mask Downsampling**: As the feature maps are downsampled through the network, the mask is also
       downsampled to maintain alignment.
    
    3. **Sparse Computation**: Convolution operations are only applied to visible (unmasked) regions,
       making the computation more efficient.
    """)
    
    # Depthwise Separable Convolution Explanation
    st.subheader("Efficient Predictor with Depthwise Separable Convolutions")
    st.write("""
    The predictor in CNN-JEPA uses depthwise separable convolutions to efficiently process the high-dimensional
    feature maps produced by the context encoder.
    """)
    
    # Create a diagram illustrating depthwise separable convolution
    fig = create_depthwise_separable_diagram()
    st.pyplot(fig)
    
    st.markdown("""
    **Depthwise Separable Convolutions:**
    
    1. **Depthwise Convolution**: Applies a separate filter to each input channel, producing one output
       channel per input channel.
    
    2. **Pointwise Convolution**: Applies a 1×1 convolution to combine the outputs from the depthwise
       convolution, producing the desired number of output channels.
    
    3. **Efficiency**: This approach significantly reduces the number of parameters and computational
       cost compared to standard convolutions, making it ideal for the predictor in CNN-JEPA.
    """)
    
    # Training Process
    st.subheader("Training Process")
    st.write("""
    CNN-JEPA is trained using a self-supervised approach:
    
    1. **Input Processing**: An image is processed through the model.
    
    2. **Masking**: A portion of the image is masked (typically ~40%).
    
    3. **Context Encoding**: The masked image is processed by the sparse context encoder.
    
    4. **Target Encoding**: The original, unmasked image is processed by the target encoder.
    
    5. **Prediction**: The predictor takes the context features and predicts the target features
       for the masked regions.
    
    6. **Loss Calculation**: The L2 distance between predicted and target features is calculated
       only at masked locations.
    
    7. **Parameter Updates**: The context encoder and predictor are updated via gradient descent,
       while the target encoder is updated via Exponential Moving Average (EMA) of the context encoder.
    """)
    
    # Performance and Results
    st.subheader("Performance and Results")
    st.write("""
    CNN-JEPA demonstrates strong performance on standard benchmarks:
    
    - **ImageNet-100**: Achieves 73.3% linear top-1 accuracy with ResNet-50, outperforming I-JEPA
      with ViT backbones.
    
    - **ImageNet-1K**: Competitive performance against established CNN-based SSL methods like BYOL,
      SimCLR, and VICReg.
    
    - **Efficiency**: Requires 17-35% less training time compared to other CNN SSL methods for the
      same number of epochs.
    
    - **Simplicity**: Achieves these results with minimal data augmentations and without a separate
      projector network.
    """)
    
    # Comparison with Other Methods
    st.subheader("Comparison with Other Methods")
    
    comparison_data = {
        "Method": ["CNN-JEPA", "I-JEPA", "BYOL", "SimCLR", "VICReg"],
        "Architecture": ["CNN", "ViT", "CNN", "CNN", "CNN"],
        "Approach": ["Predictive", "Predictive", "Invariance", "Invariance", "Invariance+Variance"],
        "Augmentations": ["Minimal", "No", "Heavy", "Heavy", "Heavy"],
        "Projector": ["No", "No", "Yes", "Yes", "Yes"],
        "Training Efficiency": ["High", "Medium", "Low", "Low", "Low"]
    }
    
    st.table(comparison_data)
    
    # Conclusion
    st.subheader("Conclusion")
    st.write("""
    CNN-JEPA represents a significant advancement in self-supervised learning for CNNs:
    
    - It successfully adapts the JEPA framework to CNNs, addressing the inherent challenges of
      using masked inputs and predictive architectures with convolutional networks.
    
    - The sparse CNN encoder and efficient predictor design are key innovations that make the
      approach work effectively with CNNs.
    
    - The method achieves strong performance on standard benchmarks while requiring less training
      time and a simpler design compared to other CNN-based SSL methods.
    
    - CNN-JEPA demonstrates that the predictive approach in latent space can be effective for
      CNNs, not just for Vision Transformers.
    """)
    
    # References
    st.subheader("References")
    st.markdown("""
    1. Dufumier, B., Bose, J., Assran, M., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). 
       *CNN-JEPA: Joint-Embedding Predictive Architecture for Convolutional Networks*. 
       arXiv preprint.
    
    2. Assran, M., Duval, F., Bose, J., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). 
       *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. 
       arXiv:2301.08243.
    
    3. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., & Piot, B. (2020).
       *Bootstrap your own latent: A new approach to self-supervised learning*.
       NeurIPS 2020.
    """)
