"""Reinforcement pool calculations for initial placement and regular turns."""

from __future__ import annotations

from .contracts import MapSpec, ReinforcementPool, TerritoryState


def calculate_reinforcement_pools(
    player_id: str,
    territories: tuple[TerritoryState, ...],
    board: MapSpec,
    *,
    pool_scope: str,
) -> tuple[ReinforcementPool, ...]:
    owned_territory_ids = {
        territory.territory_id
        for territory in territories
        if territory.owner_player_id == player_id
    }
    if not owned_territory_ids:
        raise ValueError(
            f"Cannot calculate reinforcement pools for inactive player {player_id!r}."
        )

    pools = [
        ReinforcementPool(
            pool_id=f"{player_id}:{pool_scope}:territory-count",
            player_id=player_id,
            origin="territory_count",
            remaining_armies=max(3, len(owned_territory_ids) // 2),
        )
    ]
    for continent in board.continents:
        continent_territory_ids = tuple(
            territory.territory_id
            for territory in board.territories
            if territory.continent_id == continent.continent_id
        )
        if (
            continent.reinforcement_bonus > 0
            and continent_territory_ids
            and set(continent_territory_ids).issubset(owned_territory_ids)
        ):
            pools.append(
                ReinforcementPool(
                    pool_id=(
                        f"{player_id}:{pool_scope}:continent:{continent.continent_id}"
                    ),
                    player_id=player_id,
                    origin="continent_bonus",
                    remaining_armies=continent.reinforcement_bonus,
                    allowed_territory_ids=continent_territory_ids,
                )
            )
    return tuple(pools)


__all__ = ["calculate_reinforcement_pools"]
