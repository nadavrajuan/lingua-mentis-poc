#!/usr/bin/env python3
"""Bootstrap the system: ensure model, data, and DB are ready."""
import sys
import os
sys.path.insert(0, "/app")


def ensure_dirs():
    for d in ["/app/data/mnist", "/app/models", "/app/data"]:
        os.makedirs(d, exist_ok=True)


def mnist_exists() -> bool:
    base = "/app/data/mnist/MNIST/raw"
    return os.path.exists(os.path.join(base, "t10k-images-idx3-ubyte"))


def base_model_exists() -> bool:
    return os.path.exists("/app/models/mnist_cnn_base.pt")


def ambiguity_bank_exists() -> bool:
    return os.path.exists("/app/data/ambiguity_bank.sqlite")


def download_mnist():
    print("Downloading MNIST...")
    from torchvision import datasets, transforms
    datasets.MNIST("/app/data/mnist", train=True, download=True,
                   transform=transforms.ToTensor())
    datasets.MNIST("/app/data/mnist", train=False, download=True,
                   transform=transforms.ToTensor())
    print("MNIST downloaded.")


def train_base_model():
    print("Training base model...")
    import scripts.train_base_model as t
    t.train()


def build_ambiguity_bank():
    print("Building ambiguity bank...")
    import scripts.build_ambiguity_bank as b
    b.build()


def init_experiment_db():
    import asyncio
    from app.db import init_db
    asyncio.run(init_db())
    print("Experiment DB initialized.")


def main():
    ensure_dirs()

    if not mnist_exists():
        download_mnist()
    else:
        print("MNIST data found.")

    if not base_model_exists():
        train_base_model()
    else:
        print("Base model found.")

    if not ambiguity_bank_exists():
        build_ambiguity_bank()
    else:
        print("Ambiguity bank found.")

    init_experiment_db()
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
