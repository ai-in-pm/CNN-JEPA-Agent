import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import torch

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the Python path
sys.path.append(script_dir)

from demo.cnn_jepa_demo import CNNJEPADemo
from demo.educational import show_educational_content
from demo.technical_details import show_technical_details
from utils.visualization import (
    visualize_masking, 
    visualize_predictions, 
    visualize_feature_maps, 
    visualize_embeddings
)

# Set page configuration
st.set_page_config(
    page_title="CNN-JEPA Demonstration",
    page_icon="🧠",
    layout="wide"
)

def main():
    st.title("CNN-JEPA: Joint-Embedding Predictive Architecture for CNNs")
    st.subheader("An Interactive Demonstration")

    # Sidebar with explanation and navigation
    with st.sidebar:
        st.header("About CNN-JEPA")
        st.write("""
        CNN-JEPA is a self-supervised learning method that adapts the Joint-Embedding Predictive 
        Architecture (JEPA) framework to Convolutional Neural Networks (CNNs).
        
        It introduces several key innovations to make the JEPA approach work effectively with 
        convolutional architectures, including a sparse CNN encoder, an efficient predictor using 
        depthwise separable convolutions, and a CNN-compatible masking strategy.
        """)

        st.header("Key Components")
        st.markdown("""
        - **Sparse CNN Encoder**: Handles masked inputs and produces masked feature maps
        - **Efficient Predictor**: Uses depthwise separable convolutions for parameter efficiency
        - **CNN-Compatible Masking**: Adapted for CNN downsampling
        - **Minimal Augmentations**: Requires only random resized crop
        - **No Projector Network**: Simpler architecture than many SSL methods
        """)

        st.header("Navigation")
        page = st.radio(
            "Select a page:",
            ["Interactive Demo", "Educational Content", "Technical Details"]
        )

    # Display the selected page
    if page == "Interactive Demo":
        show_interactive_demo()
    elif page == "Educational Content":
        show_educational_content()
    elif page == "Technical Details":
        show_technical_details()

def show_interactive_demo():
    """
    Display the interactive CNN-JEPA demonstration
    """
    st.header("Interactive CNN-JEPA Demo")
    
    st.write("""
    This interactive demonstration shows how CNN-JEPA processes images and predicts representations
    for masked regions. You can upload your own image or use a sample image, adjust the masking
    parameters, and see how the model processes the image.
    """)
    
    # Initialize the demo
    demo = CNNJEPADemo()
    
    # Image selection
    st.subheader("Image Selection")
    
    image_source = st.radio(
        "Select image source:",
        ["Upload Image", "Sample Image", "Generate Synthetic Image"]
    )
    
    image_path = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name
    
    elif image_source == "Sample Image":
        # Use a sample image
        sample_images = {
            "Sample 1": "https://images.unsplash.com/photo-1533450718592-29d45635f0a9?q=80&w=1000&auto=format&fit=crop",
            "Sample 2": "https://images.unsplash.com/photo-1543877087-ebf71fde2be1?q=80&w=1000&auto=format&fit=crop",
            "Sample 3": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?q=80&w=1000&auto=format&fit=crop"
        }
        
        selected_sample = st.selectbox("Select a sample image:", list(sample_images.keys()))
        
        # Download the sample image
        import requests
        from io import BytesIO
        
        response = requests.get(sample_images[selected_sample])
        img = Image.open(BytesIO(response.content))
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file, format="JPEG")
            image_path = tmp_file.name
    
    elif image_source == "Generate Synthetic Image":
        # Generate a synthetic image
        img = demo.generate_synthetic_image()
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file, format="JPEG")
            image_path = tmp_file.name
    
    # Process parameters
    st.subheader("Masking Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        masking_ratio = st.slider("Masking Ratio (%)", 10, 70, 40)
    with col2:
        block_size = st.slider("Block Size", 16, 64, 32, step=8)
    with col3:
        num_target_blocks = st.slider("Number of Target Blocks", 1, 4, 1)
    
    # Process button
    if image_path and st.button("Process Image"):
        with st.spinner("Processing image with CNN-JEPA..."):
            # Process the image
            (original_image, masked_image, context_mask, target_mask, 
             context_features, predicted_features, target_features, final_mask) = demo.process_image(
                image_path, masking_ratio=masking_ratio/100, block_size=block_size, 
                num_target_blocks=num_target_blocks
            )
            
            # Display results
            st.subheader("Results")
            
            # Masking visualization
            st.write("**Masking Visualization**")
            fig = visualize_masking(original_image, context_mask, target_mask)
            st.pyplot(fig)
            
            # Feature maps visualization
            st.write("**Feature Maps Visualization**")
            fig = visualize_feature_maps(context_features, predicted_features, target_features, final_mask)
            st.pyplot(fig)
            
            # Prediction visualization
            st.write("**Prediction Results**")
            fig = visualize_predictions(original_image, context_mask, target_mask, 
                                       predicted_features, target_features, final_mask)
            st.pyplot(fig)
            
            # Embedding visualization
            st.write("**Feature Embedding Visualization**")
            fig = visualize_embeddings(predicted_features.detach().cpu().numpy())
            st.pyplot(fig)
    
    # Quick explanation
    with st.expander("How does this demonstration work?"):
        st.write("""
        This interactive demonstration shows how CNN-JEPA processes image data:
        
        1. **Input**: An image is selected and processed
        2. **Masking**: Regions of the image are masked according to the specified parameters
        3. **Processing**: The masked image is processed through the sparse CNN encoder
        4. **Prediction**: The model predicts features for the masked regions using the depthwise separable CNN predictor
        5. **Visualization**: The results are visualized, showing the original image, masked image, feature maps, and predictions
        
        Note that this is a simplified demonstration. A full CNN-JEPA model would be trained on millions of images
        and would learn more sophisticated representations.
        
        The key insight of CNN-JEPA is that by adapting the JEPA framework to CNNs, we can achieve strong self-supervised
        learning performance with convolutional architectures, which have different inductive biases than Vision Transformers.
        """)

if __name__ == "__main__":
    main()
