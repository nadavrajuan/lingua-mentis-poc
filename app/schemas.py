from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class DynamicParams(BaseModel):
    intensity: float = 0.01
    interval_ms: int = 700

    return_to_normal_enabled: bool = False
    return_rate: float = 0.2

    confidence_mode: str = "inverse_confidence"
    inhibition_strength: float = 1.0

    max_delta_norm: float = 3.0
    max_single_update_norm: float = 0.2

    dynamic_layer_mode: str = "fc2_only"

    sequence_mode: str = "random"
    class_a: Optional[int] = None
    class_b: Optional[int] = None

    show_true_label: bool = True
    noise_std: float = 0.0


class PredictionResult(BaseModel):
    image_id: int
    true_label: int
    predicted_label: int
    probabilities: list[float]
    vanilla_probabilities: list[float] = []
    confidence: float
    second_label: int
    margin: float
    delta_norm: float
    last_update_norm: float
    image_base64: str = ""


class SessionCreateRequest(BaseModel):
    params: DynamicParams = Field(default_factory=DynamicParams)
    notes: str = ""


class ParamsUpdateRequest(BaseModel):
    params: DynamicParams


class HumanTrialResponse(BaseModel):
    user_label: int
    response_time_ms: int


class CalibrationResult(BaseModel):
    score: float
    params: DynamicParams
