# UNETR-2D-Medical-Image-Segmentation

This repository provides a **from-scratch PyTorch implementation** of a 2D variant of the **UNETR** (U-Net with Transformers) architecture for medical image segmentation. The original UNETR model was introduced by [Hatamizadeh et al. (2021)](https://arxiv.org/pdf/2103.10504.pdf) for 3D volumetric medical image segmentation tasks, leveraging Transformers as an encoder to learn global context. Here, I adapt the design to a 2D setting, serving as a conceptual reference rather than a fully trained, production-ready model.

**Note:** This implementation is a demonstration of the architecture. It is not pre-trained, and thus does not produce meaningful segmentation results without proper training and dataset preparation.

## Table of Contents
- [Overview of UNETR](#overview-of-unetr)
- [Differences from the Original Paper](#differences-from-the-original-paper)
- [Model Architecture](#model-architecture)
    - [Patch Embedding](#patch-embedding)
    - [Transformer Encoder](#transformer-encoder)
    - [CNN Decoder (U-Net like)](#cnn-decoder-u-net-like)
- [Code Structure](#code-structure)
- [Usage](#usage)

---

## Overview of UNETR

**UNETR** combines the strengths of the Transformer-based ViT (Vision Transformer) encoder with a U-Net style CNN decoder. The main idea behind UNETR is:

- Use a Transformer to encode spatial and semantic information globally from the input image patches.  
- Inject high-level semantic features at different Transformer layers directly into a convolutional decoder, similar to skip connections in a standard U-Net.  
- Benefit from both global context (Transformers) and spatial precision (CNN decoder) to enhance segmentation performance.

In the original paper, UNETR was designed for 3D medical imaging, particularly MRI scans, where volumetric data (3D cubes) are split into patches and processed by a ViT encoder. The decoder then reconstructs the segmentation map.

## Differences from the Original Paper

- **Dimensionality**: This implementation focuses on **2D images** rather than 3D volumes. The logic, however, remains similar:  
  - Instead of a 3D patch embedding, we flatten 2D patches of the input image.
  - Instead of a 3D Transformer encoder, we utilize a 2D adaptation (effectively treating each patch as a token).
  
- **No Pre-training or Training**: The code as provided does not include a training loop or pre-trained weights. It’s solely an architectural reference.

- **Testing on a Single Image**: A single test image is processed at the end of the script to visualize the model’s output. Since the model is untrained, the output mask is not meaningful.

## Model Architecture

The architecture can be divided into three main parts:

### Patch Embedding
- The input image (e.g., 256x256 with 3 channels) is divided into non-overlapping patches (e.g., 16x16).
- Each patch is flattened into a vector, then linearly projected into a hidden dimension space, serving as a "token".
- A positional embedding is added to these tokens to retain spatial information.

### Transformer Encoder
- A series of Transformer encoder layers process the embedded patches.
- Each layer consists of Multi-Head Self-Attention and MLP blocks, alongside Layer Normalization and residual connections.
- Selected Transformer layer outputs (e.g., after layers 3, 6, 9, and 12) are extracted as feature maps. These serve a role similar to skip connections in U-Net, ensuring the decoder can leverage features at different scales and depths of abstraction.

### CNN Decoder (U-Net like)
- The decoder uses transposed convolutions (and related convolutional blocks) to upsample and fuse the encoded features.
- Starting from the deepest Transformer features, the decoder progressively merges features from shallower Transformer layers and ultimately the input-level features.
- The output is a single-channel segmentation mask of the same spatial size as the input image.

## Code Structure

- **`UNETR` class**:  
  - Handles the overall model pipeline: patch embedding, positional embedding, Transformer encoder, and convolutional decoder.
  - Encapsulates logic for reshaping token embeddings back into spatial feature maps.

- **Blocks**:
  - `OrangeBlock`: Convolution + BatchNorm + ReLU block.
  - `GreenBlock`: Transpose convolution for upsampling.
  - `BlueBlock`: Upsampling followed by convolution and normalization (used in decoder stages).
  - `GreyBlock`: A simple 1x1 convolution for the final output layer.

- **Main script**:
  - Defines model configuration (image size, patch size, Transformer depth, etc.).
  - Loads a sample image, processes it into patches, and runs it through the model.
  - Visualizes the input image and the output segmentation map (untrained).

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Y1D1R/UNETR-2D-Medical-Image-Segmentation.git
   cd UNETR-2D-Medical-Image-Segmentation
   
2. **Install Requirements**:
   To install the required dependencies, run:
   ```bash
   pip install -r requirements.txt

3. **Run the script**:
   ```bash
   python UNETR-2D.py   
   
