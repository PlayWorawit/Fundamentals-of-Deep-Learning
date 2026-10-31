# CIFAR-10 Image Classification Using CNN

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset into 10 different categories.

The main objective is to build and train a deep learning model from scratch, evaluate its performance, and understand how CNNs can be applied to image classification tasks.

## 2. Dataset

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR-10 contains 60,000 color images with a resolution of 32 × 32 pixels, divided into 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Dataset Split

The project uses the following data split:

| Dataset | Number of Images |
|---|---:|
| Training | 8,000 |
| Validation | 2,000 |
| Testing | 10,000 |

The training and validation sets are selected from the official CIFAR-10 training set using a fixed random seed. The official test set is kept separate and used for final evaluation.

The dataset is automatically downloaded when running the training or evaluation scripts. It is not included in this repository.

## 3. Model Architecture

The project implements a custom CNN architecture named `ImprovedCNN`.

The model consists of:

- **Block 1:** Convolutional layers with 32 filters, Batch Normalization, ReLU, Max Pooling, and Dropout.
- **Block 2:** Convolutional layers with 64 filters, Batch Normalization, ReLU, Max Pooling, and Dropout.
- **Block 3:** Convolutional layer with 128 filters, Batch Normalization,
