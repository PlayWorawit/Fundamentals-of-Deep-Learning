# Improved CNN for Image Classification

## 1. Project Overview

This project focuses on image classification using a Convolutional Neural Network (CNN) built with PyTorch.

The project uses the CIFAR-10 dataset, which contains 10 image classes. The goal is to develop and improve a CNN model based on the starter code provided in the course.

## 2. Dataset

- **Dataset:** CIFAR-10
- **Number of classes:** 10
- **Image size:** 32 × 32 pixels
- **Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

The dataset is loaded using `torchvision`. The dataset is not included in this repository.

## 3. Model Architecture

This project implements an improved CNN model using PyTorch.

The model includes:

- Three convolutional blocks.
- Convolutional layers with 32, 64, and 128 filters.
- Batch Normalization.
- ReLU activation functions.
- Max Pooling layers.
- Dropout for regularization.
- Fully connected layers, including a hidden layer with 256 units.
- An output layer for 10 classes.

The model is developed by modifying and improving the course starter code rather than using a pretrained model directly.

## 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 128 |
| Epochs | 30 |
| Weight Decay | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Random Seed | 0 |
| Image Size | 32 × 32 |

## 5. Repository Structure

```text
Fundamentals-of-Deep-Learning/
│
├── notebooks/
│   └── DEEPLEARN.ipynb
│
└── README.md
```

## 6. How to Run the Project

The easiest way to run this project is through Google Colab.

1. Open the notebook in the `notebooks/` folder.
2. Open it using Google Colab.
3. Select a runtime environment (GPU is recommended).
4. Run the notebook cells in order.
5. The dataset will be loaded through `torchvision`.
6. Run the training and evaluation sections to reproduce the experiment.

## 7. Evaluation

The model is evaluated using test data.

The notebook includes:

- Training and validation loss.
- Training and validation accuracy.
- Test loss and test accuracy.
- Confusion matrix.
- Per-class recall.

## 8. Acknowledgements

This project was developed as part of a Fundamentals of Deep Learning assignment.

The course-provided starter code and dataset were used as the starting point. The CNN architecture was further modified and improved as part of this project.
