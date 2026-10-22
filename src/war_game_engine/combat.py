"""Pure combat resolution and exact dice-outcome distributions."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product
from typing import Annotated

from pydantic import Field

from .contracts import ContractModel

DieFace = Annotated[int, Field(ge=1, le=6)]


class CombatDiceFaces(ContractModel):
    attacker_faces: tuple[DieFace, ...] = Field(min_length=1, max_length=3)
    defender_faces: tuple[DieFace, ...] = Field(min_length=1, max_length=3)


def _resolve_valid_faces(
    attacker_faces: tuple[int, ...],
    defender_faces: tuple[int, ...],
) -> tuple[int, int]:
    attacker_rolls = sorted(attacker_faces, reverse=True)
    defender_rolls = sorted(defender_faces, reverse=True)
    attacker_losses = 0
    defender_losses = 0
    for attacker_face, defender_face in zip(attacker_rolls, defender_rolls):
        if attacker_face <= defender_face:
            attacker_losses += 1
        else:
            defender_losses += 1
    return attacker_losses, defender_losses


def resolve_combat_round(
    attacker_faces: tuple[int, ...],
    defender_faces: tuple[int, ...],
) -> tuple[int, int]:
    """Return attacker and defender losses for one set of dice results."""
    faces = CombatDiceFaces(
        attacker_faces=attacker_faces,
        defender_faces=defender_faces,
    )
    return _resolve_valid_faces(faces.attacker_faces, faces.defender_faces)


def enumerate_combat_loss_distribution(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> dict[tuple[int, int], Fraction]:
    """Enumerate exact loss probabilities for all legal six-sided dice rolls."""
    if (
        isinstance(attacker_dice_count, bool)
        or isinstance(defender_dice_count, bool)
        or not 1 <= attacker_dice_count <= 3
        or not 1 <= defender_dice_count <= 3
    ):
        raise ValueError("Combat requires one to three dice on each side.")

    total_rolls = 6 ** (attacker_dice_count + defender_dice_count)
    loss_counts: Counter[tuple[int, int]] = Counter()
    for roll in product(
        range(1, 7),
        repeat=attacker_dice_count + defender_dice_count,
    ):
        attacker_losses, defender_losses = _resolve_valid_faces(
            tuple(roll[:attacker_dice_count]),
            tuple(roll[attacker_dice_count:]),
        )
        loss_counts[(attacker_losses, defender_losses)] += 1
    return {
        losses: Fraction(count, total_rolls)
        for losses, count in sorted(loss_counts.items())
    }


__all__ = [
    "CombatDiceFaces",
    "enumerate_combat_loss_distribution",
    "resolve_combat_round",
]
