#!/usr/bin/env python3
"""Precompute confusing MNIST examples from the test set."""
import sys
sys.path.insert(0, "/app")

import json
import os
import sqlite3
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_PATH = "/app/models/mnist_cnn_base.pt"
DATA_DIR = "/app/data/mnist"
AMBIGUITY_DB = "/app/data/ambiguity_bank.sqlite"

LOW_CONFIDENCE_THRESH = 0.75
LOW_MARGIN_THRESH = 0.20


def build():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from app.model import MNISTCNN
    model = MNISTCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    conn = sqlite3.connect(AMBIGUITY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ambiguity (
            image_id INTEGER PRIMARY KEY,
            true_label INTEGER,
            predicted_label INTEGER,
            probs_json TEXT,
            confidence REAL,
            second_label INTEGER,
            margin REAL
        )
    """)
    conn.execute("DELETE FROM ambiguity")

    image_id = 0
    rows = []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            logits, _ = model(data)
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = probs.sort(dim=-1, descending=True)

            for i in range(data.size(0)):
                p = probs[i].tolist()
                confidence = float(sorted_p[i, 0].item())
                pred = int(sorted_i[i, 0].item())
                second = int(sorted_i[i, 1].item())
                margin = float(sorted_p[i, 0].item() - sorted_p[i, 1].item())
                true_label = int(targets[i].item())

                rows.append((image_id, true_label, pred, json.dumps(p), confidence, second, margin))
                image_id += 1

    conn.executemany(
        "INSERT INTO ambiguity VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()

    low_conf = conn.execute("SELECT COUNT(*) FROM ambiguity WHERE confidence < ?", (LOW_CONFIDENCE_THRESH,)).fetchone()[0]
    low_margin = conn.execute("SELECT COUNT(*) FROM ambiguity WHERE margin < ?", (LOW_MARGIN_THRESH,)).fetchone()[0]
    print(f"Total images: {image_id}")
    print(f"Low confidence (<{LOW_CONFIDENCE_THRESH}): {low_conf}")
    print(f"Low margin (<{LOW_MARGIN_THRESH}): {low_margin}")
    conn.close()
    print(f"Ambiguity bank saved to {AMBIGUITY_DB}")


if __name__ == "__main__":
    if os.path.exists(AMBIGUITY_DB):
        print(f"Ambiguity bank already exists at {AMBIGUITY_DB}, skipping.")
    else:
        build()
