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

The model consists of three convolutional blocks and a fully connected classifier.

- **Block 1:** Two convolutional layers with 32 filters, Batch Normalization, ReLU, Max Pooling, and Dropout.
- **Block 2:** Two convolutional layers with 64 filters, Batch Normalization, ReLU, Max Pooling, and Dropout.
- **Block 3:** One convolutional layer with 128 filters, Batch Normalization, ReLU, Max Pooling, and Dropout.
- **Classifier:** Flatten layer, fully connected layer with 256 neurons, ReLU, Dropout, and output layer with 10 neurons.

The model uses Batch Normalization to stabilize training and Dropout to help reduce overfitting.

## 4. Training Configuration

The model is trained using the following configuration:

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Model | ImprovedCNN |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.0005 |
| Weight Decay | 0.0001 |
| Batch Size | 128 |
| Number of Epochs | 30 |
| Random Seed | 0 |
| Number of Classes | 10 |

### Data Preprocessing and Augmentation

The training images are preprocessed using:

- Random Horizontal Flip
- Random Crop with padding of 4 pixels
- Conversion to Tensor
- Normalization using CIFAR-10 mean and standard deviation

For validation and testing, images are converted to tensors and normalized without applying random augmentation.

## 5. Repository Structure

```text
Fundamentals-of-Deep-Learning/
│
├── notebooks/
│   └── DEEPLEARN.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### File Descriptions

| File | Description |
|---|---|
| `notebooks/DEEPLEARN.ipynb` | Jupyter Notebook containing the project implementation |
| `src/model.py` | Defines the ImprovedCNN architecture |
| `src/train.py` | Handles model training and saves the best checkpoint |
| `src/evaluate.py` | Evaluates the trained model on the test dataset |
| `requirements.txt` | Lists the required Python libraries |
| `.gitignore` | Excludes datasets, model checkpoints, and unnecessary files |

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/PlayWorawit/Fundamentals-of-Deep-Learning.git
```

Navigate to the project directory:

```bash
cd Fundamentals-of-Deep-Learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 7. Training the Model

To train the CNN model, run:

```bash
python src/train.py
```

The training script will:

1. Download the CIFAR-10 dataset if it is not available.
2. Prepare the training and validation datasets.
3. Initialize the ImprovedCNN model.
4. Train the model for 30 epochs.
5. Evaluate the model on the validation set after each epoch.
6. Save the best model checkpoint.

The best checkpoint is saved at:

```text
models/best_model.pth
```

The `models/` directory is created automatically during training.

## 8. Model Evaluation

After training, evaluate the model using the official CIFAR-10 test set:

```bash
python src/evaluate.py
```

The evaluation script reports:

- Test Loss
- Test Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class classification performance

The evaluation script uses the best saved model checkpoint generated during training.

## 9. Reproducibility

This project uses a fixed random seed (`SEED = 0`) to improve reproducibility.

The random seed is applied to:

- Python random
- NumPy
- PyTorch

The training and validation datasets are selected using a seeded random permutation.

The official CIFAR-10 test set is kept separate from training and validation to prevent data leakage.

The model architecture, hyperparameters, and training configuration are documented in this repository.

## 10. Experimental Results

The model was trained for 30 epochs using the CIFAR-10 dataset.

### Training and Validation Results

| Metric | Result |
|---|---:|
| Training Accuracy | 74.98% |
| Validation Accuracy | 71.95% |
| Best Validation Loss | 0.778 |

### Test Results

The final model was evaluated on the official CIFAR-10 test set containing 10,000 images.

| Metric | Result |
|---|---:|
| Test Loss | 0.7874 |
| Test Accuracy | 72.19% |
| Macro Precision | 72.39% |
| Macro Recall | 72.19% |
| Macro F1-score | 71.87% |

### Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Airplane | 78.79% | 73.20% | 75.89% |
| Automobile | 80.92% | 91.20% | 85.75% |
| Bird | 69.12% | 52.60% | 59.74% |
| Cat | 54.04% | 43.50% | 48.20% |
| Deer | 69.44% | 62.50% | 65.79% |
| Dog | 53.55% | 74.00% | 62.13% |
| Frog | 77.36% | 78.60% | 77.98% |
| Horse | 72.83% | 79.60% | 76.06% |
| Ship | 83.01% | 86.50% | 84.72% |
| Truck | 84.87% | 80.20% | 82.47% |

### Discussion

The model achieved a test accuracy of **72.19%**, demonstrating its ability to classify images across the 10 CIFAR-10 categories.

Automobile achieved the highest recall at 91.20%, while Cat had the lowest recall at 43.50%. This indicates that the model performs better on some classes than others.

Further improvements could explore different CNN architectures, data augmentation techniques, and hyperparameter configurations.

## 11. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Google Colab

## 12. Author

**PlayWorawit**

GitHub: [PlayWorawit](https://github.com/PlayWorawit)
