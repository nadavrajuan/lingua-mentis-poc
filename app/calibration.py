from __future__ import annotations
import json
import math
import itertools
import torch
import torch.nn.functional as F
from app.model import MNISTCNN
from app.dynamic_engine import DynamicState, predict
from app.schemas import DynamicParams
from app.data import get_image_tensor

INTENSITY_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
RETURN_RATES = [0.0, 0.05, 0.2, 0.5, 1.0]
INHIBITION_VALUES = [0.5, 1.0, 1.5]
CONFIDENCE_MODES = ["inverse_confidence", "binned"]


def run_calibration(trials: list[dict], base_model: MNISTCNN) -> tuple[float, DynamicParams]:
    best_score = float("inf")
    best_params = DynamicParams()

    grid = list(itertools.product(
        INTENSITY_VALUES, RETURN_RATES, INHIBITION_VALUES, CONFIDENCE_MODES
    ))

    for intensity, return_rate, inhibition, conf_mode in grid:
        params = DynamicParams(
            intensity=intensity,
            return_rate=return_rate,
            inhibition_strength=inhibition,
            confidence_mode=conf_mode,
            return_to_normal_enabled=return_rate > 0,
        )
        score = _evaluate(trials, base_model, params)
        if score < best_score:
            best_score = score
            best_params = params

    return best_score, best_params


def _evaluate(trials: list[dict], base_model: MNISTCNN, params: DynamicParams) -> float:
    state = DynamicState.create("calibration", base_model, params)
    total_loss = 0.0
    n = 0

    for trial in trials:
        user_label = trial.get("user_label")
        if user_label is None:
            continue
        image_id = trial["image_id"]
        true_label = trial["true_label"]

        try:
            tensor, _ = _get_tensor(image_id)
            result = predict(state, tensor, image_id, true_label)
        except Exception:
            continue

        prob_human = result.probabilities[user_label]
        loss = -math.log(max(prob_human, 1e-9))
        total_loss += loss
        n += 1

    if n == 0:
        return float("inf")

    stability_penalty = max(0.0, state.delta_norm() - params.max_delta_norm) * 0.5
    return total_loss / n + stability_penalty


def _get_tensor(image_id: int) -> tuple[torch.Tensor, int]:
    return get_image_tensor(image_id)
