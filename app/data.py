from __future__ import annotations
import io
import base64
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageFilter

DATA_DIR = "/app/data/mnist"
_test_dataset: datasets.MNIST | None = None


def get_test_dataset() -> datasets.MNIST:
    global _test_dataset
    if _test_dataset is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        _test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return _test_dataset


def get_image_tensor(image_id: int) -> tuple[torch.Tensor, int]:
    ds = get_test_dataset()
    img_tensor, label = ds[image_id]
    return img_tensor.unsqueeze(0), int(label)


def tensor_to_base64(image_id: int, degrade_params: dict | None = None) -> str:
    ds = get_test_dataset()
    raw_img, _ = ds.data[image_id], ds.targets[image_id]
    pil = Image.fromarray(raw_img.numpy(), mode="L")
    if degrade_params:
        pil = _degrade(pil, degrade_params)
    pil = pil.resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _degrade(img: Image.Image, params: dict) -> Image.Image:
    if "downsample_to" in params:
        size = params["downsample_to"]
        img = img.resize((size, size), Image.BILINEAR).resize((28, 28), Image.BILINEAR)
    if "blur_radius" in params:
        img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
    if "noise_std" in params:
        rng = np.random.RandomState(params.get("seed", 0))
        arr = np.array(img).astype(float)
        arr += rng.normal(0, params["noise_std"] * 255, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    if "contrast" in params:
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(params["contrast"])
    return img
