"""Sequence models over kill windows.

Two shapes of model, because the labels and the evidence live at different levels:

    FlickCNN   scores one window. Simple, and what the pipeline runs per moment.
    PlayerMIL  scores a player from several of their windows at once. The label
               is a player, and most of a cheater's kills look ordinary, so the
               model is allowed to decide which of their windows matter. This is
               multiple-instance learning: a bag of windows, one label.

Pooling is where a sequence becomes a verdict:

    avgmax     mean and max over time, concatenated. Cheap, hard to beat.
    attention  the model learns which ticks to weight. Slightly better, and the
               weights are readable — useful later for showing a reviewer *when*
               in the window the model objected.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Weighted average over the last axis, with learned weights.

    Returns the pooled vector and the weights, so the caller can show which ticks
    (or which windows) drove the score.
    """

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, channels, steps) -> (batch, steps, channels)
        steps = x.transpose(1, 2)
        weights = torch.softmax(self.score(steps).squeeze(-1), dim=-1)
        pooled = torch.einsum("bsc,bs->bc", steps, weights)
        return pooled, weights


class FlickCNN(nn.Module):
    """A 1D convolutional net over the tick axis of one window."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        *,
        pooling: str = "avgmax",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width * 2, width * 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(width * 3),
            nn.GELU(),
        )
        self.embedding_size = width * 3 * (2 if pooling == "avgmax" else 1)
        self.attention = AttentionPool(width * 3) if pooling == "attention" else None
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.embedding_size, 1)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """One vector per window, before the final decision."""
        h = self.body(x)
        if self.attention is not None:
            pooled, _ = self.attention(h)
            return pooled
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)


class PlayerMIL(nn.Module):
    """Score a player from a bag of their windows.

    Each window is encoded by the same FlickCNN body, then attention decides how
    much each window counts. A player who cheated in three kills out of twenty
    can still be scored highly without the other seventeen dragging them down.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        *,
        window_pooling: str = "avgmax",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = FlickCNN(
            in_channels, width, pooling=window_pooling, dropout=dropout
        )
        size = self.encoder.embedding_size
        self.attention = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, 1))

    def forward(
        self, bags: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            bags: (players, windows, channels, ticks)
            mask: (players, windows), 1 for a real window, 0 for padding.

        Returns:
            (logit per player, attention weight per window)
        """
        players, windows = bags.shape[:2]
        flat = bags.reshape(players * windows, *bags.shape[2:])
        embedded = self.encoder.embed(flat).reshape(players, windows, -1)

        scores = self.attention(embedded).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        pooled = torch.einsum("pwe,pw->pe", embedded, weights)
        return self.head(pooled).squeeze(-1), weights


class _ReverseGradient(torch.autograd.Function):
    """Identity going forward; flips the gradient going back."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * grad, None


def reverse_gradient(x: torch.Tensor, strength: float) -> torch.Tensor:
    return _ReverseGradient.apply(x, strength)


class InvariantFlickCNN(nn.Module):
    """A window scorer trained to ignore which weapon was used.

    Cheaters in CS2CD take 60% of their kills with snipers and clean players 15%,
    so a model can score well by recognising sniper play — and then it flags clean
    AWP mains. Fixing the visibility measurement did not help, because the leak is
    in the data rather than the instrument.

    So a second head tries to predict the weapon class from the same features, and
    a gradient-reversal layer makes the encoder *worse* at that task while staying
    good at the first one. What survives is what predicts cheating without
    revealing the weapon.

    `strength` is the trade-off: 0 reproduces the plain model, and too high costs
    real accuracy, since weapon and behaviour are genuinely entangled.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        *,
        n_weapon_classes: int = 3,
        pooling: str = "avgmax",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = FlickCNN(in_channels, width, pooling=pooling, dropout=dropout)
        size = self.encoder.embedding_size
        self.weapon_head = nn.Sequential(
            nn.Linear(size, 64), nn.GELU(), nn.Linear(64, n_weapon_classes)
        )

    def forward(
        self, x: torch.Tensor, strength: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder.embed(x)
        cheat = self.encoder.head(features).squeeze(-1)
        weapon = self.weapon_head(reverse_gradient(features, strength))
        return cheat, weapon
