"""Motion-adaptive hierarchical anomaly network and its training objective."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import log
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


_MOTION_INDICES = (0, 1, 5, 6, 9, 10, 13, 14)
_SHAPE_INDICES = (2, 3, 4, 7, 8, 11, 12, 15, 16)
_DIRECTION_INDICES = (5, 6, 9, 10)


@dataclass(frozen=True)
class ModelConfig:
    num_scenes: int = 29
    pose_input_dim: int = 85
    motion_input_dim: int = 31
    pose_hidden_dim: int = 96
    motion_hidden_dim: int = 48
    shape_hidden_dim: int = 32
    scene_embedding_dim: int = 16
    router_hidden_dim: int = 32
    node_hidden_dim: int = 64
    dropout: float = 0.1
    probability_epsilon: float = 1e-6
    robust_scales: tuple[float, ...] = (1.0,) * 12
    clip_upper: tuple[float, ...] = (10.0,) * 12
    initial_feature_weights: tuple[float, ...] = (1.0 / 12,) * 12
    # Retained checkpoint buffer; not used to route the three free experts.
    cluster_temperature: float = 0.35
    max_context_log_weight_adjustment: float = 0.35
    max_orientation_log_weight_adjustment: float = 0.35
    orientation_reliability_mode: str = "continuous"
    free_soft_usage_min: float = 0.10
    free_soft_usage_max: float = 0.75
    free_hard_usage_min: float = 0.05
    free_hard_usage_max: float = 0.90
    free_soft_usage_weight: float = 0.015
    free_hard_usage_weight: float = 0.030
    cluster_temporal_weight: float = 0.005


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    probabilities: Tensor
    prediction_mask: Tensor
    relative_motion_intensity: Tensor
    state_boundaries: Tensor
    state_probabilities: Tensor
    root_early_probability: Tensor
    root_continue_probability: Tensor
    state_early_probability: Tensor
    state_continue_probability: Tensor
    pose_anomaly_probability: Tensor
    pose_reach_probability: Tensor
    route_leaf_probabilities: Tensor
    node_anomaly_contributions: Tensor
    pose_embedding: Tensor
    motion_embedding: Tensor
    shape_embedding: Tensor


class _TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.recurrent = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True,
            bidirectional=True,
        )

    def forward(self, values: Tensor, lengths: Tensor, valid: Tensor) -> Tensor:
        projected = self.projection(values * valid.unsqueeze(-1))
        projected = projected * valid.unsqueeze(-1)
        packed = pack_padded_sequence(
            projected, lengths.detach().cpu(), batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=values.shape[1],
        )
        return encoded * valid.unsqueeze(-1)


def _node_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
    )


class _ZeroContext(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.register_buffer("_dtype_anchor", torch.empty(0), persistent=False)

    def forward(self, scene_ids: Tensor) -> Tensor:
        return torch.zeros(
            (*scene_ids.shape, self.dimension), device=scene_ids.device,
            dtype=self._dtype_anchor.dtype,
        )


def _preserve_initialization_stream(config: ModelConfig) -> None:
    """Preserve seeded initialization draws without unused model layers.

    Temporary layers preserve the checkpoint-compatible initialization stream.
    They never enter the network, its parameter count, or an optimizer.
    """
    nn.Embedding(config.num_scenes, config.scene_embedding_dim)
    context = config.scene_embedding_dim + 4
    hidden = config.router_hidden_dim
    nn.Sequential(nn.Linear(context, hidden), nn.GELU(), nn.Linear(hidden, 4))
    nn.Sequential(nn.Linear(context, hidden), nn.GELU(), nn.Linear(hidden, 2))


class _MotionRouter(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.register_buffer("robust_scales", torch.as_tensor(config.robust_scales, dtype=torch.float32).clone())
        self.register_buffer("clip_upper", torch.as_tensor(config.clip_upper, dtype=torch.float32).clone())
        self.register_buffer("cluster_temperature", torch.tensor(float(config.cluster_temperature), dtype=torch.float32))
        self.global_weight_raw = nn.Parameter(torch.tensor(
            [log(torch.expm1(torch.tensor(float(v), dtype=torch.float64)).item())
             for v in config.initial_feature_weights], dtype=torch.float32,
        ))
        hidden = config.router_hidden_dim
        self.context_weight_network = nn.Sequential(
            nn.Linear(config.scene_embedding_dim + 4, hidden), nn.GELU(),
            nn.Linear(hidden, 12),
        )
        nn.init.zeros_(self.context_weight_network[-1].weight)
        nn.init.zeros_(self.context_weight_network[-1].bias)
        self.max_context_log_weight_adjustment = float(config.max_context_log_weight_adjustment)
        self.orientation_weight_network = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 12),
        )
        nn.init.zeros_(self.orientation_weight_network[-1].weight)
        nn.init.zeros_(self.orientation_weight_network[-1].bias)
        self.max_orientation_log_weight_adjustment = float(config.max_orientation_log_weight_adjustment)
        self.orientation_reliability_mode = config.orientation_reliability_mode
        self.free_gate = nn.Sequential(
            nn.Linear(config.motion_input_dim, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )

    def _orientation_values(self, values: Tensor) -> tuple[Tensor, Tensor]:
        orientation = values[..., 29:31]
        if not torch.isfinite(orientation).all() or ((orientation < -1e-7) | (orientation > 1 + 1e-7)).any():
            raise ValueError("orientation score and reliability must be finite in [0, 1]")
        orientation = orientation.clamp(0, 1)
        return orientation[..., 0], orientation[..., 1]

    def _component_weights(self, values: Tensor, context_zeros: Tensor) -> Tensor:
        signed = values[..., _DIRECTION_INDICES]
        direction = signed / signed.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.cat((context_zeros, direction), dim=-1)
        adjustment = self.max_context_log_weight_adjustment * torch.tanh(self.context_weight_network(context))
        global_weights = F.softplus(self.global_weight_raw).clamp_min(1e-6)
        weights = global_weights * torch.exp(adjustment)
        base_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        score, reliability = self._orientation_values(values)
        orientation_context = torch.stack((2.0 * score - 1.0, reliability), dim=-1)
        raw = self.orientation_weight_network(orientation_context)
        adjustment = self.max_orientation_log_weight_adjustment * torch.tanh(raw) * reliability.unsqueeze(-1)
        adjusted = base_weights * torch.exp(adjustment)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neutral = (adjustment == 0).all(dim=-1, keepdim=True)
        exact_base_with_adjusted_gradient = adjusted + (base_weights - adjusted).detach()
        weights = torch.where(neutral, exact_base_with_adjusted_gradient, adjusted)
        return torch.where(values[..., 30].unsqueeze(-1) == 0, base_weights, weights)

    def forward(self, values: Tensor, context_zeros: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._orientation_values(values)
        evidence = values[..., 17:29]
        if not torch.isfinite(evidence).all() or (evidence < -1e-7).any():
            raise ValueError("motion evidence must be finite and non-negative")
        normalized = torch.minimum(torch.log1p(evidence.clamp_min(0) / self.robust_scales), self.clip_upper)
        intensity = (self._component_weights(values, context_zeros) * normalized).sum(-1)
        gate_input = torch.cat((values[..., :17], normalized, values[..., 29:]), -1)
        probabilities = torch.softmax(self.free_gate(gate_input), -1)
        # Output compatibility only: free experts have no ordered boundaries.
        return intensity, intensity.new_zeros((*intensity.shape, 2)), probabilities


def _bounded_usage_penalty(usage: Tensor, lower: float, upper: float) -> Tensor:
    below = F.relu((float(lower) - usage) / float(lower)).square()
    above = F.relu((usage - float(upper)) / (1.0 - float(upper))).square()
    return (below + above).mean()


class MotionAdaptiveNetwork(nn.Module):
    """Three free experts with hierarchical motion, shape, and pose decisions.

    ``scene_ids`` retains the batching interface only. No scene values enter
    the network. The zero context preserves the published head dimensions.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.pose_input_dim != 85 or cfg.motion_input_dim != 31 or cfg.scene_embedding_dim != 16:
            raise ValueError("the released model uses P85, M31, and a 16-D zero context")
        if cfg.num_scenes < 1 or not 0 < cfg.probability_epsilon < .5:
            raise ValueError("invalid model configuration")
        if cfg.orientation_reliability_mode != "continuous":
            raise ValueError("the released model uses continuous orientation reliability")
        for name in ("robust_scales", "clip_upper", "initial_feature_weights"):
            values = torch.as_tensor(getattr(cfg, name))
            if values.shape != (12,) or not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{name} must contain twelve finite positive values")
        for lower, upper in ((cfg.free_soft_usage_min, cfg.free_soft_usage_max),
                             (cfg.free_hard_usage_min, cfg.free_hard_usage_max)):
            if not 0 < lower < upper < 1:
                raise ValueError("usage bounds must satisfy 0 < lower < upper < 1")
        self.pose_encoder = _TemporalEncoder(cfg.pose_input_dim, cfg.pose_hidden_dim, cfg.dropout)
        self.motion_encoder = _TemporalEncoder(len(_MOTION_INDICES), cfg.motion_hidden_dim, cfg.dropout)
        self.shape_encoder = _TemporalEncoder(len(_SHAPE_INDICES), cfg.shape_hidden_dim, cfg.dropout)
        _preserve_initialization_stream(cfg)
        self.scene_embedding = _ZeroContext(cfg.scene_embedding_dim)
        # Reserve this registration position before initializing the heads.
        self.state_router = nn.Identity()
        root_dim = 2 * cfg.motion_hidden_dim + cfg.scene_embedding_dim + 1
        shape_dim = 2 * cfg.shape_hidden_dim + cfg.scene_embedding_dim + 1
        pose_dim = 2 * cfg.pose_hidden_dim + cfg.scene_embedding_dim + 1
        self.root_early_head = _node_mlp(root_dim, cfg.node_hidden_dim, cfg.dropout)
        self.still_shape_head = _node_mlp(shape_dim, cfg.node_hidden_dim, cfg.dropout)
        self.slow_shape_head = _node_mlp(shape_dim, cfg.node_hidden_dim, cfg.dropout)
        self.fast_motion_head = _node_mlp(root_dim + 1, cfg.node_hidden_dim, cfg.dropout)
        self.pose_heads = nn.ModuleList([_node_mlp(pose_dim, cfg.node_hidden_dim, cfg.dropout) for _ in range(3)])
        for head in (self.root_early_head, self.still_shape_head, self.slow_shape_head, self.fast_motion_head):
            nn.init.constant_(head[-1].bias, -2.)
        self.state_router = _MotionRouter(cfg)

    @staticmethod
    def _validate_prefix_mask(mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1)
        if (lengths < 1).any():
            raise ValueError("every sequence must contain a valid time step")
        expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(mask, expected):
            raise ValueError("sequence_mask must be prefix-valid padding")
        return lengths

    def forward(self, pose_features: Tensor, motion_features: Tensor,
                scene_ids: Tensor, sequence_mask: Tensor, *,
                pose_valid: Tensor | None = None,
                motion_valid: Tensor | None = None) -> ModelOutput:
        cfg = self.config
        if pose_features.ndim != 3 or pose_features.shape[-1] != cfg.pose_input_dim:
            raise ValueError("pose_features has an unexpected shape")
        if motion_features.ndim != 3 or motion_features.shape[-1] != cfg.motion_input_dim:
            raise ValueError("motion_features has an unexpected shape")
        if pose_features.shape[:2] != motion_features.shape[:2]:
            raise ValueError("pose and motion batch/time dimensions must match")
        batch, time = pose_features.shape[:2]
        device = pose_features.device
        if scene_ids.shape != (batch,):
            raise ValueError("scene_ids must have shape [batch]")
        scene_ids = scene_ids.to(device=device, dtype=torch.long)
        if ((scene_ids < 0) | (scene_ids >= cfg.num_scenes)).any():
            raise ValueError("scene_ids are outside the configured vocabulary")
        sequence_mask = sequence_mask.to(device=device, dtype=torch.bool)
        if sequence_mask.shape != (batch, time):
            raise ValueError("sequence_mask must have shape [batch, time]")
        lengths = self._validate_prefix_mask(sequence_mask)
        pose_valid = sequence_mask if pose_valid is None else pose_valid.to(device=device, dtype=torch.bool) & sequence_mask
        motion_valid = sequence_mask if motion_valid is None else motion_valid.to(device=device, dtype=torch.bool) & sequence_mask
        if pose_valid.shape != (batch, time) or motion_valid.shape != (batch, time):
            raise ValueError("branch validity masks must have shape [batch, time]")
        pose_encoded = self.pose_encoder(pose_features, lengths, pose_valid)
        motion_encoded = self.motion_encoder(motion_features[..., _MOTION_INDICES], lengths, motion_valid)
        shape_encoded = self.shape_encoder(motion_features[..., _SHAPE_INDICES], lengths, motion_valid)
        scene_encoded = self.scene_embedding(scene_ids).unsqueeze(1).expand(-1, time, -1)
        intensity, boundaries, state_probabilities = self.state_router(motion_features * motion_valid.unsqueeze(-1), scene_encoded)
        motion_flag = motion_valid.to(pose_features.dtype).unsqueeze(-1)
        pose_flag = pose_valid.to(pose_features.dtype).unsqueeze(-1)
        root_input = torch.cat((motion_encoded, scene_encoded, motion_flag), dim=-1)
        root_early = torch.sigmoid(self.root_early_head(root_input).squeeze(-1)) * motion_flag.squeeze(-1)
        root_continue = 1.0 - root_early
        shape_input = torch.cat((shape_encoded, scene_encoded, motion_flag), dim=-1)
        fast_input = torch.cat((root_input, intensity.unsqueeze(-1)), dim=-1)
        state_early = torch.stack((
            torch.sigmoid(self.still_shape_head(shape_input).squeeze(-1)),
            torch.sigmoid(self.slow_shape_head(shape_input).squeeze(-1)),
            torch.sigmoid(self.fast_motion_head(fast_input).squeeze(-1)),
        ), dim=-1) * motion_flag
        state_continue = 1.0 - state_early
        pose_input = torch.cat((pose_encoded, scene_encoded, pose_flag), dim=-1)
        pose_anomaly = torch.stack([torch.sigmoid(head(pose_input).squeeze(-1)) for head in self.pose_heads], dim=-1) * pose_flag
        state_mass = root_continue.unsqueeze(-1) * state_probabilities
        early_mass = state_mass * state_early
        pose_reach_mass = state_mass * state_continue
        pose_anomaly_mass = pose_reach_mass * pose_anomaly
        pose_normal_mass = pose_reach_mass * (1.0 - pose_anomaly)
        leaves, anomaly_contributions = [root_early], [root_early]
        for index in range(3):
            leaves.extend((early_mass[..., index], pose_normal_mass[..., index], pose_anomaly_mass[..., index]))
            anomaly_contributions.extend((early_mass[..., index], pose_anomaly_mass[..., index]))
        route_leaf_probabilities = torch.stack(leaves, dim=-1)
        node_anomaly_contributions = torch.stack(anomaly_contributions, dim=-1)
        probabilities = node_anomaly_contributions.sum(dim=-1)
        prediction_mask = sequence_mask & (pose_valid | motion_valid)
        active = prediction_mask.to(pose_features.dtype)
        probabilities = probabilities * active
        route_leaf_probabilities = route_leaf_probabilities * active.unsqueeze(-1)
        node_anomaly_contributions = node_anomaly_contributions * active.unsqueeze(-1)
        eps = cfg.probability_epsilon
        logits = (torch.log(probabilities.clamp_min(eps)) - torch.log1p(-probabilities.clamp_max(1.0 - eps))) * active
        return ModelOutput(
            logits=logits, probabilities=probabilities, prediction_mask=prediction_mask,
            relative_motion_intensity=intensity * active,
            state_boundaries=boundaries * active.unsqueeze(-1),
            state_probabilities=state_probabilities * active.unsqueeze(-1),
            root_early_probability=root_early * active,
            root_continue_probability=root_continue * active,
            state_early_probability=state_early * active.unsqueeze(-1),
            state_continue_probability=state_continue * active.unsqueeze(-1),
            pose_anomaly_probability=pose_anomaly * active.unsqueeze(-1),
            pose_reach_probability=pose_reach_mass.sum(dim=-1) * active,
            route_leaf_probabilities=route_leaf_probabilities,
            node_anomaly_contributions=node_anomaly_contributions,
            pose_embedding=pose_encoded, motion_embedding=motion_encoded,
            shape_embedding=shape_encoded,
        )

    def auxiliary_training_loss(self, output: ModelOutput, active: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        valid = active.bool() & output.prediction_mask & (output.state_early_probability.sum(-1) > 0)
        zero = output.logits.sum() * 0
        if not valid.any():
            return zero, {"free_balance_active_count": valid.sum()}
        cfg = self.config
        probabilities = output.state_probabilities[valid]
        soft_usage = probabilities.mean(0)
        hard = F.one_hot(probabilities.argmax(-1), num_classes=3).to(probabilities.dtype)
        hard_st = hard + probabilities - probabilities.detach()
        hard_usage = hard_st.mean(0)
        soft_guard = _bounded_usage_penalty(soft_usage, cfg.free_soft_usage_min, cfg.free_soft_usage_max)
        hard_guard = _bounded_usage_penalty(hard_usage, cfg.free_hard_usage_min, cfg.free_hard_usage_max)
        adjacent = valid[:, 1:] & valid[:, :-1]
        temporal = F.smooth_l1_loss(output.relative_motion_intensity[:, 1:][adjacent],
                                  output.relative_motion_intensity[:, :-1][adjacent]) if adjacent.any() else zero
        loss = cfg.free_soft_usage_weight * soft_guard + cfg.free_hard_usage_weight * hard_guard + cfg.cluster_temporal_weight * temporal
        return loss, {
            "free_balance_active_count": valid.sum(),
            "free_balance_soft_usage": soft_usage.detach(),
            "free_balance_hard_usage": hard.detach().mean(0),
            "free_balance_soft_guard": soft_guard.detach(),
            "free_balance_hard_guard": hard_guard.detach(),
            "free_balance_intensity_temporal": temporal.detach(),
        }


def make_model(config: Mapping[str, Any] | None = None) -> MotionAdaptiveNetwork:
    """Build from current fields, ignoring training/preprocessing metadata."""
    accepted = {field.name for field in fields(ModelConfig)}
    return MotionAdaptiveNetwork(ModelConfig(**{key: value for key, value in (config or {}).items() if key in accepted}))


def forward_batch(model: nn.Module, batch: Mapping[str, Any]) -> ModelOutput:
    return model(batch["pose_features"], batch["motion_features"], batch["scene_ids"],
                 batch["sequence_mask"], pose_valid=batch["pose_valid"],
                 motion_valid=batch["motion_valid"])
