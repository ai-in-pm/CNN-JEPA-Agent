# CNN-JEPA Demonstration

This project provides an interactive demonstration of CNN-JEPA (Convolutional Neural Network Joint-Embedding Predictive Architecture), a PhD-level Artificial Machine Intelligence (AMI) that showcases how CNN-JEPA works for learning visual representations from image data.
![CNN-JEPA TechDetails](https://github.com/user-attachments/assets/34722609-63e2-4751-a478-37ce34bb9d60)
![CNN-JEPA SyntheticImageGen](https://github.com/user-attachments/assets/8659a082-6c73-4f29-9be4-adcd77878c2e)
![CNN-JEPA SampleImage](https://github.com/user-attachments/assets/2bdedc5e-c8c1-445a-9c43-64662e5e5ed7)
![CNN-JEPA SampleImage III](https://github.com/user-attachments/assets/ef7ce2ad-f3df-4635-ae17-dabed77e873b)
![CNN-JEPA SampleImage II](https://github.com/user-attachments/assets/0ea4b70b-7d56-4e9a-9eb0-5cf783191821)

## Overview

CNN-JEPA is a novel self-supervised learning method that adapts the Joint-Embedding Predictive Architecture (JEPA) framework, previously successful with Vision Transformers (ViTs), to Convolutional Neural Networks (CNNs). It addresses the inherent challenges of using masked inputs and predictive architectures with CNNs through several key innovations.

This demonstration provides:
1. An interactive interface to explore CNN-JEPA's masking and prediction capabilities
2. Educational content explaining the key concepts and innovations
3. Technical details about the implementation and architecture

## Key Components

- **Sparse CNN Encoder**: Modified CNN architecture that can handle masked inputs and produce masked feature maps
- **Efficient Predictor**: Fully convolutional predictor using depthwise separable convolutions for parameter efficiency
- **CNN-Compatible Masking**: Masking strategy adapted for CNN downsampling
- **Minimal Augmentations**: Requires only random resized crop, unlike many SSL methods
- **No Projector Network**: Simpler architecture than many other SSL methods

## Features of this Demonstration

- **Interactive Masking**: Adjust masking ratio, block size, and number of target blocks
- **Real-time Processing**: Process images through the CNN-JEPA model in real-time
- **Visualization**: Visualize masking strategy, feature maps, predictions, and embeddings
- **Educational Content**: Learn about the key concepts and innovations of CNN-JEPA
- **Technical Details**: Explore the implementation details and architecture

## Installation

```bash
# Clone the repository (if applicable)
# git clone https://github.com/your-username/cnn-jepa-demo.git
# cd cnn-jepa-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the demonstration:

```bash
python run_demo.py
```

This will:
1. Start the Streamlit app
2. Open the demonstration in your browser

Additional options:

```bash
# Run on a specific port
python run_demo.py --port 8502

# Run in debug mode
python run_demo.py --debug
```

## Project Structure

```
CNN-JEPA-Agent/
├── app.py                 # Main Streamlit application
├── run_demo.py            # Script to run the demonstration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── demo/                  # Demo components
│   ├── cnn_jepa_demo.py   # Core demonstration logic
│   ├── educational.py     # Educational content
│   └── technical_details.py # Technical details
├── models/                # Model architecture
│   └── cnn_jepa_model.py  # CNN-JEPA model implementation
├── utils/                 # Utility functions
│   ├── masking.py         # Masking strategies
│   └── visualization.py   # Visualization utilities
```

## References

1. Dufumier, B., Bose, J., Assran, M., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). *CNN-JEPA: Joint-Embedding Predictive Architecture for Convolutional Networks*. arXiv preprint.

2. Assran, M., Duval, F., Bose, J., Misra, I., Bojanowski, P., Joulin, A., Brock, A., Rabbat, M., & Synnaeve, G. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243.

3. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., & Piot, B. (2020). *Bootstrap your own latent: A new approach to self-supervised learning*. NeurIPS 2020.
