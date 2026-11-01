
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from model import ImprovedCNN


# ==========================================
# 1. Configuration
# ==========================================

DATA_DIR = "./data"
MODEL_DIR = Path("./models")
RESULTS_DIR = Path("./results")

BATCH_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXPERIMENTS = [
    "baseline_dropout",
    "ablation_no_dropout"
]


# ==========================================
# 2. Prepare Test Dataset
# ==========================================

norm = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

test_tf = transforms.Compose([
    transforms.ToTensor(),
    norm
])


def prepare_test_loader():

    test_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_tf
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    classes = test_set.classes

    print("Test samples:", len(test_set))
    print("Classes:", classes)

    return test_loader, classes


# ==========================================
# 3. Evaluate One Model
# ==========================================

def evaluate_model(experiment_name, test_loader, classes):

    model_path = MODEL_DIR / f"{experiment_name}.pth"

    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        return None

    print("\n" + "=" * 60)
    print("Evaluating:", experiment_name)
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # Read dropout configuration
    use_dropout = checkpoint["config"]["use_dropout"]

    model = ImprovedCNN(
        n_classes=len(classes),
        use_dropout=use_dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []

    # Test loop
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Calculate metrics
    test_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(classes))),
            average=None,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # Print overall results
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall:    {recall * 100:.2f}%")
    print(f"Macro F1-score:  {f1 * 100:.2f}%")

    # Print per-class results
    print("\nPer-Class Results")
    print("-" * 75)

    print(
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )

    for i, class_name in enumerate(classes):

        print(
            f"{class_name:<15}"
            f"{class_precision[i] * 100:>11.2f}%"
            f"{class_recall[i] * 100:>11.2f}%"
            f"{class_f1[i] * 100:>11.2f}%"
            f"{class_support[i]:>12}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    result = {
        "experiment": experiment_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {}
    }

    for i, class_name in enumerate(classes):

        result["per_class"][class_name] = {
            "precision": float(class_precision[i]),
            "recall": float(class_recall[i]),
            "f1": float(class_f1[i]),
            "support": int(class_support[i])
        }

    return result


# ==========================================
# 4. Compare Both Models
# ==========================================

if __name__ == "__main__":

    print("Device:", device)

    test_loader, classes = prepare_test_loader()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for experiment_name in EXPERIMENTS:

        result = evaluate_model(
            experiment_name,
            test_loader,
            classes
        )

        if result is not None:
            all_results[experiment_name] = result

    # Print comparison
    print("\n" + "=" * 75)
    print("FINAL TEST SET COMPARISON")
    print("=" * 75)

    print(
        f"{'Experiment':<25}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
    )

    for name, result in all_results.items():

        print(
            f"{name:<25}"
            f"{result['test_accuracy'] * 100:>11.2f}%"
            f"{result['macro_precision'] * 100:>11.2f}%"
            f"{result['macro_recall'] * 100:>11.2f}%"
            f"{result['macro_f1'] * 100:>11.2f}%"
        )

    # Save results
    output_path = RESULTS_DIR / "test_comparison.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nEvaluation completed.")
    print("Results saved at:", output_path)
