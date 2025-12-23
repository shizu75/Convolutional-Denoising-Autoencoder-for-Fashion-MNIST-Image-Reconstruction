# Convolutional Autoencoder for Fashion-MNIST Image Reconstruction

## Project Overview
This project presents a **research-grade Convolutional Autoencoder (CAE)** implemented using **TensorFlow/Keras** for **unsupervised image representation learning** on the **Fashion-MNIST dataset**. The model learns compact latent representations of grayscale clothing images and reconstructs them with minimal information loss. This work demonstrates core deep learning concepts used in **computer vision, anomaly detection, feature compression, and medical imaging pipelines**, making it suitable for **graduate-level research portfolios (MS/PhD)**.

---

## Dataset Description
- **Dataset:** Fashion-MNIST
- **Source:** Keras built-in dataset
- **Image Resolution:** 28 × 28 pixels
- **Color Channels:** Grayscale (1 channel)
- **Classes:** 10 (labels ignored – unsupervised learning)
- **Training Samples:** 50,000 (after split)
- **Validation Samples:** 10,000
- **Test Samples:** 10,000
- **Pixel Normalization:** Rescaled to [0, 1]

---

## Problem Statement
High-dimensional image data contains redundant spatial information. The objective is to:
- Learn **compressed latent representations** using convolutional encoders
- Reconstruct original images using decoders
- Minimize reconstruction loss without using class labels

---

## Methodology

### Data Preprocessing
- Images normalized using `Rescaling(1/255)`
- Converted to `float32`
- Reshaped to `(28, 28, 1)` for convolutional processing
- Dataset split into training, validation, and test subsets

---

## Model Architecture

### Encoder
- Conv2D (32 filters, 3×3, ReLU, same padding)
- MaxPooling2D (2×2)
- Conv2D (16 filters, 3×3, ReLU, same padding)
- MaxPooling2D (2×2)
- Conv2D (8 filters, 3×3, ReLU, same padding)
- MaxPooling2D (2×2)

This progressively reduces spatial dimensions while increasing feature abstraction.

### Decoder
- Conv2D (8 filters, 3×3, ReLU, same padding)
- UpSampling2D (2×2)
- Conv2D (16 filters, 3×3, ReLU, same padding)
- UpSampling2D (2×2)
- Conv2D (32 filters, 3×3, ReLU)
- UpSampling2D (2×2)
- Conv2D (1 filter, 3×3, Sigmoid, same padding)

The decoder reconstructs the image back to its original resolution.

---

## Training Configuration
- **Framework:** TensorFlow / Keras Sequential API
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Epochs:** Up to 50
- **Callbacks:** EarlyStopping (patience = 5, monitored on validation loss)
- **Training Objective:** Minimize reconstruction error

---

## Evaluation and Results
- Reconstruction loss evaluated on unseen test data
- Training and validation loss curves plotted for convergence analysis
- Visual comparison between:
  - Original Fashion-MNIST image
  - Autoencoder-reconstructed image

The model successfully reconstructs clothing silhouettes, preserving structural patterns while smoothing fine details—typical behavior of convolutional autoencoders.

---

## Key Observations
- Effective spatial feature compression using convolutional layers
- Stable training with early stopping preventing overfitting
- Latent representations capture meaningful visual structures
- Suitable baseline for anomaly detection and representation learning

---

## Research Relevance
This project demonstrates:
- Practical application of **unsupervised deep learning**
- Convolutional feature extraction without supervision
- Foundations for advanced models such as:
  - Variational Autoencoders (VAEs)
  - Denoising Autoencoders
  - Self-supervised learning frameworks

It is directly relevant to research in **computer vision, biomedical imaging, signal processing, and AI-driven diagnostics**.

---

## Technologies Used
- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

---

## Potential Extensions
- Latent space visualization using PCA or t-SNE
- Denoising autoencoder with noisy inputs
- Quantitative evaluation using SSIM / PSNR
- Transfer learning using encoded representations
- Application to medical or industrial image datasets

---

## Academic Note
This repository is structured and documented to meet **research portfolio standards**, emphasizing reproducibility, clarity, and extensibility for **graduate admissions and research evaluation**.
