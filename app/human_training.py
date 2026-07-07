from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from app.data import get_image_tensor, tensor_to_base64
from app.sampler import _class_index
from app.model import MNISTCNN
import torch
import torch.nn.functional as F


@dataclass
class HumanSession:
    session_id: str
    mode: str
    class_a: int | None
    class_b: int | None
    degrade_params: dict | None
    trial_index: int = 0
    current_image_id: int = -1
    current_true_label: int = -1
    sequence: list[int] = field(default_factory=list)

    @classmethod
    def create(cls, session_id: str, mode: str, class_a: int | None, class_b: int | None,
               degrade_params: dict | None) -> "HumanSession":
        seq = _build_sequence(mode, class_a, class_b)
        inst = cls(session_id=session_id, mode=mode, class_a=class_a, class_b=class_b,
                   degrade_params=degrade_params, sequence=seq)
        return inst

    def advance(self) -> tuple[int, int] | None:
        if self.trial_index >= len(self.sequence):
            return None
        self.current_image_id = self.sequence[self.trial_index]
        from app.data import get_test_dataset
        ds = get_test_dataset()
        self.current_true_label = int(ds.targets[self.current_image_id].item())
        self.trial_index += 1
        return self.current_image_id, self.current_true_label


def _build_sequence(mode: str, class_a: int | None, class_b: int | None, n: int = 100) -> list[int]:
    idx = _class_index()
    total = sum(len(v) for v in idx.values())

    if mode == "random":
        return random.sample(range(total), min(n, total))
    elif mode == "3_vs_8":
        pool = idx.get(3, []) + idx.get(8, [])
        random.shuffle(pool)
        return pool[:n]
    elif mode == "custom_pair":
        pool = []
        if class_a is not None:
            pool.extend(idx.get(class_a, []))
        if class_b is not None:
            pool.extend(idx.get(class_b, []))
        random.shuffle(pool)
        return pool[:n]
    else:
        all_ids = list(range(total))
        random.shuffle(all_ids)
        return all_ids[:n]


def get_base_prediction(model: MNISTCNN, image_id: int) -> tuple[int, list[float]]:
    tensor, _ = get_image_tensor(image_id)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    pred = int(torch.tensor(probs).argmax().item())
    return pred, probs


def compute_human_summary(trials: list[dict]) -> dict:
    total = len(trials)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "confusion": {}}

    correct = sum(1 for t in trials if t.get("user_label") == t["true_label"])
    confusion: dict[str, int] = {}
    for t in trials:
        key = f"{t['true_label']}->{t.get('user_label', '?')}"
        confusion[key] = confusion.get(key, 0) + 1

    avg_rt = (
        sum(t["response_time_ms"] for t in trials if t.get("response_time_ms"))
        / max(1, sum(1 for t in trials if t.get("response_time_ms")))
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "avg_response_time_ms": avg_rt,
        "confusion": confusion,
    }
