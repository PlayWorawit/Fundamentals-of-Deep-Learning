
import os
import random
import json
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
RESULTS_DIR = Path("./results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Experiments
# ==========================================

EXPERIMENTS = [
    {
        "name": "baseline_dropout",
        "use_dropout": True
    },
    {
        "name": "ablation_no_dropout",
        "use_dropout": False
    }
]


# ==========================================
# 3. Reproducibility
# ==========================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 4. Dataset Preparation
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

    test_set = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=eval_tf
    )

    # Same seeded split for every experiment
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(full_train),
        generator=generator
    )[:SUBSET_SIZE].tolist()

    n_val = SUBSET_SIZE // 5

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_set = Subset(full_train, train_indices)

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
# 5. Evaluation Function
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
# 6. Training Function
# ==========================================

def train_model(
    model,
    train_loader,
    val_loader,
    classes,
    experiment_name,
    use_dropout
):

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each experiment has its own checkpoint
    model_path = MODEL_DIR / f"{experiment_name}.pth"

    for epoch in range(EPOCHS):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Training and validation metrics
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

        # Save best checkpoint using validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss

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
                    "weight_decay": WEIGHT_DECAY,
                    "use_dropout": use_dropout,
                    "experiment_name": experiment_name
                }
            }, model_path)

            print("  Best model checkpoint saved.")

    # Save training history for later comparison
    history_path = RESULTS_DIR / f"{experiment_name}_history.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved at: {model_path}")
    print(f"History saved at: {history_path}")

    return history


# ==========================================
# 7. Run All Experiments
# ==========================================

if __name__ == "__main__":

    print("Device:", device)

    train_loader, val_loader, test_loader, classes = prepare_data()

    all_results = {}

    for experiment in EXPERIMENTS:

        experiment_name = experiment["name"]
        use_dropout = experiment["use_dropout"]

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Dropout enabled: {use_dropout}")
        print("=" * 60)

        # Reset seed to make experiments comparable
        set_seed(SEED)

        model = ImprovedCNN(
            n_classes=len(classes),
            use_dropout=use_dropout
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=classes,
            experiment_name=experiment_name,
            use_dropout=use_dropout
        )

        all_results[experiment_name] = {
            "best_val_acc": max(history["val_acc"]),
            "best_val_loss": min(history["val_loss"]),
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1]
        }

    # ======================================
    # 8. Experiment Comparison
    # ======================================

    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    for name, result in all_results.items():

        print(f"\n{name}")
        print(f"Best Validation Accuracy: {result['best_val_acc']:.2%}")
        print(f"Best Validation Loss:     {result['best_val_loss']:.4f}")
        print(f"Final Training Accuracy:  {result['final_train_acc']:.2%}")
        print(f"Final Validation Accuracy:{result['final_val_acc']:.2%}")

    summary_path = RESULTS_DIR / "ablation_summary.json"

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nAll experiments completed.")
    print(f"Summary saved at: {summary_path}")
