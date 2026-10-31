
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model import ImprovedCNN

# Configuration
SEED = 0
BATCH_SIZE = 128
DATA_DIR = "./data"
MODEL_PATH = "./models/best_model.pth"

# Set random seed
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10 test dataset
test_set = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = test_set.classes

# Check model checkpoint
import os

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model checkpoint not found. Run src/train.py first."
    )

# Load trained model
try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=device)

model = ImprovedCNN(n_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Evaluate model on test set
criterion = nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = total_loss / total
test_accuracy = 100.0 * correct / total

print("\n===== TEST RESULTS =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4
))
