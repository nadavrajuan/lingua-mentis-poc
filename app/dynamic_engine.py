from __future__ import annotations
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from app.model import MNISTCNN
from app.schemas import DynamicParams, PredictionResult


@dataclass
class DynamicState:
    session_id: str
    base_model: MNISTCNN
    delta_fc2_weight: torch.Tensor
    delta_fc2_bias: torch.Tensor
    delta_fc1_weight: torch.Tensor
    delta_fc1_bias: torch.Tensor
    last_update_time: float = field(default_factory=time.time)
    params: DynamicParams = field(default_factory=DynamicParams)
    last_update_norm: float = 0.0

    @classmethod
    def create(cls, session_id: str, base_model: MNISTCNN, params: DynamicParams) -> "DynamicState":
        fc2w = base_model.fc2.weight
        fc2b = base_model.fc2.bias
        fc1w = base_model.fc1.weight
        fc1b = base_model.fc1.bias
        return cls(
            session_id=session_id,
            base_model=base_model,
            delta_fc2_weight=torch.zeros_like(fc2w),
            delta_fc2_bias=torch.zeros_like(fc2b),
            delta_fc1_weight=torch.zeros_like(fc1w),
            delta_fc1_bias=torch.zeros_like(fc1b),
            params=params,
        )

    def reset_delta(self):
        self.delta_fc2_weight.zero_()
        self.delta_fc2_bias.zero_()
        self.delta_fc1_weight.zero_()
        self.delta_fc1_bias.zero_()
        self.last_update_norm = 0.0
        self.last_update_time = time.time()

    def delta_norm(self) -> float:
        return float(torch.norm(self.delta_fc2_weight).item() + torch.norm(self.delta_fc2_bias).item())


def _confidence_factor(confidence: float, mode: str) -> float:
    if mode == "inverse_confidence":
        return 1.0 - confidence
    elif mode == "margin_based":
        return 1.0 - confidence
    elif mode == "binned":
        if confidence >= 0.90:
            return 0.15
        elif confidence >= 0.75:
            return 0.40
        elif confidence >= 0.60:
            return 0.75
        else:
            return 1.00
    else:  # fixed
        return 1.0


def apply_decay(state: DynamicState):
    if not state.params.return_to_normal_enabled:
        return
    now = time.time()
    elapsed = now - state.last_update_time
    decay = math.exp(-state.params.return_rate * elapsed)
    state.delta_fc2_weight.mul_(decay)
    state.delta_fc2_bias.mul_(decay)
    state.delta_fc1_weight.mul_(decay)
    state.delta_fc1_bias.mul_(decay)
    state.last_update_time = now


def predict(state: DynamicState, image_tensor: torch.Tensor, image_id: int, true_label: int) -> PredictionResult:
    apply_decay(state)

    with torch.no_grad():
        h = state.base_model.forward_features(image_tensor)

        # vanilla: base fc2 with no delta
        vanilla_logits = F.linear(h, state.base_model.fc2.weight, state.base_model.fc2.bias)
        vanilla_probs = F.softmax(vanilla_logits, dim=-1).squeeze(0)

        mode = state.params.dynamic_layer_mode
        if mode == "fc1_fc2":
            eff_fc1_w = state.base_model.fc1.weight + state.delta_fc1_weight
            eff_fc1_b = state.base_model.fc1.bias + state.delta_fc1_bias
            h = F.relu(F.linear(image_tensor.view(image_tensor.size(0), -1), eff_fc1_w, eff_fc1_b))

        eff_fc2_w = state.base_model.fc2.weight + state.delta_fc2_weight
        eff_fc2_b = state.base_model.fc2.bias + state.delta_fc2_bias
        logits = F.linear(h, eff_fc2_w, eff_fc2_b)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    probs_list = probs.tolist()
    vanilla_probs_list = vanilla_probs.tolist()
    predicted_label = int(probs.argmax().item())
    confidence = float(probs[predicted_label].item())

    sorted_indices = probs.argsort(descending=True)
    second_label = int(sorted_indices[1].item())
    margin = float(probs[sorted_indices[0]].item() - probs[sorted_indices[1]].item())

    _update_delta(state, h.squeeze(0), probs, predicted_label)

    return PredictionResult(
        image_id=image_id,
        true_label=true_label,
        predicted_label=predicted_label,
        probabilities=probs_list,
        vanilla_probabilities=vanilla_probs_list,
        confidence=confidence,
        second_label=second_label,
        margin=margin,
        delta_norm=state.delta_norm(),
        last_update_norm=state.last_update_norm,
    )


def _update_delta(state: DynamicState, h: torch.Tensor, probs: torch.Tensor, y_hat: int):
    p = state.params
    confidence = float(probs[y_hat].item())
    cf = _confidence_factor(confidence, p.confidence_mode)

    one_hot = torch.zeros_like(probs)
    one_hot[y_hat] = 1.0

    direction = one_hot - p.inhibition_strength * probs
    direction[y_hat] = 1.0 - probs[y_hat]

    h_norm = h / (torch.norm(h) + 1e-8)

    update_matrix = torch.outer(direction, h_norm)
    update_bias = direction

    scale = p.intensity * cf

    single_norm = float(torch.norm(update_matrix).item()) * scale
    if single_norm > p.max_single_update_norm:
        scale = scale * p.max_single_update_norm / (single_norm + 1e-8)

    state.delta_fc2_weight.add_(update_matrix * scale)
    state.delta_fc2_bias.add_(update_bias * scale)
    state.last_update_norm = float(torch.norm(update_matrix * scale).item())

    current_norm = state.delta_norm()
    if current_norm > p.max_delta_norm:
        factor = p.max_delta_norm / current_norm
        state.delta_fc2_weight.mul_(factor)
        state.delta_fc2_bias.mul_(factor)
