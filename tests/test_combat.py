from collections import Counter
from fractions import Fraction
from itertools import product

import pytest

from war_game_engine.combat import (
    enumerate_combat_loss_distribution,
    resolve_combat_round,
)


def _independent_loss_distribution(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> dict[tuple[int, int], Fraction]:
    losses: Counter[tuple[int, int]] = Counter()
    for roll in product(
        range(1, 7),
        repeat=attacker_dice_count + defender_dice_count,
    ):
        attacker_rolls = sorted(roll[:attacker_dice_count], reverse=True)
        defender_rolls = sorted(roll[attacker_dice_count:], reverse=True)
        attacker_loss_count = 0
        defender_loss_count = 0
        for index in range(min(attacker_dice_count, defender_dice_count)):
            if attacker_rolls[index] <= defender_rolls[index]:
                attacker_loss_count += 1
            else:
                defender_loss_count += 1
        losses[(attacker_loss_count, defender_loss_count)] += 1

    total_rolls = 6 ** (attacker_dice_count + defender_dice_count)
    return {
        loss_pair: Fraction(count, total_rolls)
        for loss_pair, count in sorted(losses.items())
    }


def test_manual_example_resolves_ordering_and_ties_for_defense() -> None:
    assert resolve_combat_round((5, 4, 1), (6, 3, 1)) == (2, 1)
    assert resolve_combat_round((4,), (4,)) == (1, 0)


def test_unpaired_dice_do_not_cause_losses() -> None:
    assert resolve_combat_round((6, 5, 4), (6,)) == (1, 0)


@pytest.mark.parametrize("attacker_dice_count", range(1, 4))
@pytest.mark.parametrize("defender_dice_count", range(1, 4))
def test_exact_loss_distribution_matches_independent_exhaustive_resolver(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> None:
    distribution = enumerate_combat_loss_distribution(
        attacker_dice_count,
        defender_dice_count,
    )

    assert distribution == _independent_loss_distribution(
        attacker_dice_count,
        defender_dice_count,
    )
    assert sum(distribution.values(), Fraction()) == Fraction(1)


@pytest.mark.parametrize(
    ("attacker_dice_count", "defender_dice_count"),
    ((0, 1), (1, 0), (4, 1), (1, 4)),
)
def test_loss_distribution_rejects_illegal_dice_counts(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> None:
    with pytest.raises(ValueError, match="one to three dice"):
        enumerate_combat_loss_distribution(
            attacker_dice_count,
            defender_dice_count,
        )
