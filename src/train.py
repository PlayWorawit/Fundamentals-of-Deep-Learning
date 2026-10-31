import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ImprovedCNN


# ==========================================
# 1. Configuration
# ==========================================

SEED = 0
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
SUBSET_SIZE = 10000

DATA_DIR = "./data"
MODEL_DIR = Path("./models")
MODEL_PATH = MODEL_DIR / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================
# 3. Dataset Preparation
# ==========================================

norm = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    norm
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    norm
])


def prepare_data():

    full_train = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=train_tf
    )

    # Official test set remains untouched during training
    test_set = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=eval_tf
    )

    # Reproduce the notebook's seeded 10,000-image subset
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(full_train),
        generator=generator
    )[:SUBSET_SIZE].tolist()

    n_val = SUBSET_SIZE // 5

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_set = Subset(full_train, train_indices)

    # Validation data must not use training augmentation
    val_base = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=eval_tf
    )

    val_set = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    classes = full_train.classes

    print(f"Training samples:   {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples:       {len(test_set)}")
    print(f"Number of classes:  {len(classes)}")

    return train_loader, val_loader, test_loader, classes


# ==========================================
# 4. Evaluation Function
# ==========================================

@torch.no_grad()
def evaluate(model, loader, loss_fn):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


# ==========================================
# 5. Training Function
# ==========================================

def train_model(model, train_loader, val_loader):

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate using training and validation data
        train_loss, train_acc = evaluate(
            model, train_loader, loss_fn
        )

        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        # Save the best checkpoint based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            os.makedirs("models", exist_ok=True)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "config": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY
                }
            }, MODEL_PATH)

            print("  Best model checkpoint saved.")

    return history


# ==========================================
# 6. Main
# ==========================================

if __name__ == "__main__":

    print("Device:", device)

    train_loader, val_loader, test_loader, classes = prepare_data()

    model = ImprovedCNN(n_classes=len(classes))

    history = train_model(
        model,
        train_loader,
        val_loader
    )

    print("\nTraining completed.")
    print(f"Best model saved at: {MODEL_PATH}")
