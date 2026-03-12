"""Training script for EmotionCNN model."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.EmotionDataset import EmotionDataset
from .CNN import EmotionCNN

IMG_SIZE = 48
TRAIN_PATH = "dataset/train"
CLASSES = ["negative", "neutral", "positive"]


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def compute_class_weights(dataset):
    """Compute class weights to handle class imbalance."""
    class_counts = [0] * len(CLASSES)
    for _, label in dataset.samples:
        class_counts[label] += 1

    print(f"Class distribution: {dict(zip(CLASSES, class_counts))}")

    total = sum(class_counts)
    weights = [total / (len(CLASSES) * count) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=50,
    early_stopping_patience=10,
    model_path="models/cnn/best_model.pth",
):
    """Train the model with early stopping and learning rate scheduling."""
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            leave=False,
        )
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(
                loss=train_loss / (train_pbar.n + 1),
                acc=f"{100 * train_correct / train_total:.1f}%",
            )

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
            leave=False,
        )
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix(
                    loss=val_loss / (val_pbar.n + 1),
                    acc=f"{100 * val_correct / val_total:.1f}%",
                )

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train EmotionCNN model")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0). Ex: --sample 0.1 for 10%%",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Training EmotionCNN")
    print("=" * 60)

    # Device setup
    device = get_device()
    print(f"\nUsing device: {device}")

    # Data transforms - grayscale, normalize, tensor
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Load dataset
    print(f"\nLoading dataset from: {TRAIN_PATH}")
    dataset = EmotionDataset(
        TRAIN_PATH, CLASSES, transform=train_transform, sample_ratio=args.sample
    )

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(dataset).to(device)
    print(f"Class weights: {class_weights}")

    # Split train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize model
    model = EmotionCNN(num_classes=3, dropout=0.3).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: Adam with weight decay
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    model_path = "models/cnn/best_model.pth"
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=50,
        early_stopping_patience=10,
        model_path=model_path,
    )

    print("\n" + "=" * 60)
    print(f"Training complete! Best model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
