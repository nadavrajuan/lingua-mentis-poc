#!/usr/bin/env python3
"""Train the base MNIST CNN model. Runs once; skips if model already exists."""
import sys
import os
sys.path.insert(0, "/app")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "/app/data/mnist"
MODEL_PATH = "/app/models/mnist_cnn_base.pt"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    from app.model import MNISTCNN
    model = MNISTCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == target).sum().item()

        acc = correct / len(train_ds)
        print(f"Epoch {epoch + 1}/5 | loss={total_loss / len(train_loader):.4f} | acc={acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping training.")
    else:
        train()
