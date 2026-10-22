"""Typed objective catalog and deterministic win-condition evaluation."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from .board_spec import WAR1_BOARD_SPEC
from .contracts import (
    BoardSpec,
    ContractModel,
    FullGameState,
    FullPlayerState,
    MapSpec,
    NonNegativeInt,
    PositiveInt,
)

PlayerColor: TypeAlias = Literal["white", "red", "black", "blue", "yellow", "green"]

ELIMINATION_OBJECTIVE_COLORS: dict[str, PlayerColor] = {
    "W1-OBJ-01": "red",
    "W1-OBJ-02": "blue",
    "W1-OBJ-03": "white",
    "W1-OBJ-04": "black",
    "W1-OBJ-05": "yellow",
    "W1-OBJ-06": "green",
}
FALLBACK_OBJECTIVE_ID = "W1-OBJ-13"


class EliminateColorCondition(ContractModel):
    condition_type: Literal["eliminate_color"]
    target_color: PlayerColor


class ControlContinentsCondition(ContractModel):
    condition_type: Literal["control_continents"]
    required_continent_ids: tuple[str, ...] = Field(min_length=1)
    additional_continent_count: NonNegativeInt = 0


class TerritoryCountCondition(ContractModel):
    condition_type: Literal["territory_count"]
    minimum_territories: PositiveInt
    minimum_armies_per_territory: PositiveInt = 1


ObjectiveCondition: TypeAlias = Annotated[
    EliminateColorCondition
    | ControlContinentsCondition
    | TerritoryCountCondition,
    Field(discriminator="condition_type"),
]

OBJECTIVE_CONDITIONS: dict[str, ObjectiveCondition] = {
    "W1-OBJ-01": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="red",
    ),
    "W1-OBJ-02": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="blue",
    ),
    "W1-OBJ-03": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="white",
    ),
    "W1-OBJ-04": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="black",
    ),
    "W1-OBJ-05": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="yellow",
    ),
    "W1-OBJ-06": EliminateColorCondition(
        condition_type="eliminate_color",
        target_color="green",
    ),
    "W1-OBJ-07": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("north_america", "oceania"),
    ),
    "W1-OBJ-08": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("asia", "south_america"),
    ),
    "W1-OBJ-09": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("asia", "africa"),
    ),
    "W1-OBJ-10": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("north_america", "africa"),
    ),
    "W1-OBJ-11": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("europe", "south_america"),
        additional_continent_count=1,
    ),
    "W1-OBJ-12": ControlContinentsCondition(
        condition_type="control_continents",
        required_continent_ids=("europe", "oceania"),
        additional_continent_count=1,
    ),
    "W1-OBJ-13": TerritoryCountCondition(
        condition_type="territory_count",
        minimum_territories=24,
    ),
    "W1-OBJ-14": TerritoryCountCondition(
        condition_type="territory_count",
        minimum_territories=18,
        minimum_armies_per_territory=2,
    ),
}


class InvalidObjectiveStateError(ValueError):
    """Raised when an assigned objective cannot be evaluated from game state."""


def effective_objective_id(player: FullPlayerState) -> str:
    objective_id = player.private.objective_id
    if objective_id is None:
        raise InvalidObjectiveStateError(
            f"Active player {player.public.player_id!r} has no objective."
        )
    condition = OBJECTIVE_CONDITIONS.get(objective_id)
    if condition is None:
        raise InvalidObjectiveStateError(
            f"Unknown objective ID {objective_id!r} for player "
            f"{player.public.player_id!r}."
        )
    fallback_id = player.private.objective_fallback_id
    if fallback_id is not None:
        if (
            fallback_id != FALLBACK_OBJECTIVE_ID
            or not isinstance(condition, EliminateColorCondition)
        ):
            raise InvalidObjectiveStateError(
                f"Invalid fallback {fallback_id!r} for objective {objective_id!r}."
            )
        return fallback_id
    if isinstance(condition, EliminateColorCondition) and (
        condition.target_color == player.public.color
    ):
        raise InvalidObjectiveStateError(
            f"Self-targeting objective {objective_id!r} for player "
            f"{player.public.player_id!r} has no fallback."
        )
    return objective_id


def objective_is_satisfied(
    state: FullGameState,
    player: FullPlayerState,
    *,
    board: MapSpec | BoardSpec = WAR1_BOARD_SPEC,
) -> bool:
    objective_id = effective_objective_id(player)
    if (
        isinstance(board, BoardSpec)
        and objective_id not in board.objective_deck.card_ids
    ):
        raise InvalidObjectiveStateError(
            f"Objective {objective_id!r} is absent from BoardSpec."
        )
    condition = OBJECTIVE_CONDITIONS[objective_id]
    if isinstance(condition, EliminateColorCondition):
        target_player = next(
            (
                candidate
                for candidate in state.players
                if candidate.public.color == condition.target_color
            ),
            None,
        )
        if target_player is None:
            raise InvalidObjectiveStateError(
                f"Objective {objective_id!r} targets absent color "
                f"{condition.target_color!r}."
            )
        return target_player.public.is_eliminated

    if isinstance(condition, TerritoryCountCondition):
        qualified_territory_counts: dict[str, int] = {}
        for territory in state.territories:
            if (
                territory.owner_player_id is not None
                and territory.army_count
                >= condition.minimum_armies_per_territory
            ):
                qualified_territory_counts[territory.owner_player_id] = (
                    qualified_territory_counts.get(territory.owner_player_id, 0) + 1
                )
        return (
            qualified_territory_counts.get(player.public.player_id, 0)
            >= condition.minimum_territories
        )

    if isinstance(condition, ControlContinentsCondition):
        map_spec = board.map if isinstance(board, BoardSpec) else board
        map_continent_ids = {continent.continent_id for continent in map_spec.continents}
        unknown_continents = sorted(
            set(condition.required_continent_ids) - map_continent_ids
        )
        if unknown_continents:
            raise InvalidObjectiveStateError(
                f"Objective {objective_id!r} references unknown continents: "
                f"{', '.join(unknown_continents)}."
            )
        territories_by_id = {
            territory.territory_id: territory for territory in state.territories
        }
        missing_territories = sorted(
            {territory.territory_id for territory in map_spec.territories}
            - territories_by_id.keys()
        )
        if missing_territories:
            raise InvalidObjectiveStateError(
                "Objective evaluation requires the complete territory state; "
                f"missing={', '.join(missing_territories)}."
            )
        controlled_continents: set[str] = set()
        for continent in map_spec.continents:
            continent_territory_ids = tuple(
                territory.territory_id
                for territory in map_spec.territories
                if territory.continent_id == continent.continent_id
            )
            if not continent_territory_ids:
                raise InvalidObjectiveStateError(
                    f"Continent {continent.continent_id!r} has no territories."
                )
            if all(
                territories_by_id[territory_id].owner_player_id
                == player.public.player_id
                for territory_id in continent_territory_ids
            ):
                controlled_continents.add(continent.continent_id)
        required = set(condition.required_continent_ids)
        return required.issubset(controlled_continents) and (
            len(controlled_continents - required)
            >= condition.additional_continent_count
        )

    raise InvalidObjectiveStateError(
        f"Unsupported objective condition for {objective_id!r}."
    )


def objective_winners(
    state: FullGameState,
    *,
    board: MapSpec | BoardSpec = WAR1_BOARD_SPEC,
) -> tuple[str, ...]:
    return tuple(
        player.public.player_id
        for player in state.players
        if not player.public.is_eliminated
        and objective_is_satisfied(state, player, board=board)
    )


__all__ = [
    "ControlContinentsCondition",
    "EliminateColorCondition",
    "FALLBACK_OBJECTIVE_ID",
    "InvalidObjectiveStateError",
    "OBJECTIVE_CONDITIONS",
    "ObjectiveCondition",
    "PlayerColor",
    "TerritoryCountCondition",
    "effective_objective_id",
    "objective_is_satisfied",
    "objective_winners",
    "ELIMINATION_OBJECTIVE_COLORS",
]
