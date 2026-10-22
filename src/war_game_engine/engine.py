"""Pure validation and transitions for the currently implemented game phases."""

from __future__ import annotations

from typing import Literal, TypeAlias

from .board_spec import WAR1_BOARD_SPEC
from .contracts import (
    Action,
    ArmyPlacement,
    ClaimTerritoryOperation,
    FullGameState,
    FullPlayerState,
    MapSpec,
    PlaceArmiesOperation,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from .versioning import ACTION_SCHEMA_VERSION

BoardLike: TypeAlias = MapSpec


class InvalidActionError(ValueError):
    """Raised when an action is not legal for the current state."""


class UnsupportedPhaseError(ValueError):
    """Raised when a phase has not yet received its transition implementation."""


def _current_decision_player_id(state: FullGameState) -> str | None:
    return state.decision_player_id or state.active_player_id


def _territory_by_id(state: FullGameState) -> dict[str, TerritoryState]:
    return {territory.territory_id: territory for territory in state.territories}


def _pools_for_player(
    state: FullGameState, player_id: str
) -> tuple[ReinforcementPool, ...]:
    return tuple(
        pool for pool in state.reinforcement_pools if pool.player_id == player_id
    )


def _eligible_pools_for_placement(
    state: FullGameState,
    player_id: str,
    territory_id: str,
    source_pool_id: str | None,
) -> tuple[ReinforcementPool, ...]:
    pools = tuple(
        pool
        for pool in _pools_for_player(state, player_id)
        if pool.allowed_territory_ids is None
        or territory_id in pool.allowed_territory_ids
    )
    if source_pool_id is not None:
        selected = tuple(pool for pool in pools if pool.pool_id == source_pool_id)
        if not selected:
            raise InvalidActionError(
                f"Pool {source_pool_id!r} is not available for territory "
                f"{territory_id!r}."
            )
        return selected
    if len(pools) != 1:
        if not pools:
            raise InvalidActionError(
                f"No reinforcement pool can place armies on {territory_id!r}."
            )
        raise InvalidActionError(
            "source_pool_id is required when multiple reinforcement pools are eligible."
        )
    return pools


def _validate_place_armies(
    state: FullGameState,
    action: Action,
    operation: PlaceArmiesOperation,
) -> None:
    expected_phase = {
        "initial_placement": "initial",
        "reinforcement_placement": "reinforcement",
    }.get(state.phase_id)
    if expected_phase is None or operation.placement_phase != expected_phase:
        raise InvalidActionError(
            f"Placement phase {operation.placement_phase!r} is not legal during "
            f"{state.phase_id!r}."
        )

    territories = _territory_by_id(state)
    placement_territories = tuple(
        placement.territory_id for placement in operation.placements
    )
    if len(placement_territories) != len(set(placement_territories)):
        raise InvalidActionError(
            "A placement action may list each territory only once."
        )

    remaining_by_pool = {
        pool.pool_id: pool.remaining_armies
        for pool in _pools_for_player(state, action.actor_player_id)
    }
    pool_definitions = {
        pool.pool_id: pool for pool in _pools_for_player(state, action.actor_player_id)
    }
    placed_by_pool: dict[str, int] = {}
    for placement in operation.placements:
        territory = territories.get(placement.territory_id)
        if territory is None:
            raise InvalidActionError(
                f"Unknown territory ID {placement.territory_id!r}."
            )
        if territory.owner_player_id != action.actor_player_id:
            raise InvalidActionError(
                f"Player {action.actor_player_id!r} does not own territory "
                f"{placement.territory_id!r}."
            )
        pools = _eligible_pools_for_placement(
            state,
            action.actor_player_id,
            placement.territory_id,
            placement.source_pool_id,
        )
        pool = pools[0]
        placed_by_pool[pool.pool_id] = (
            placed_by_pool.get(pool.pool_id, 0) + placement.army_count
        )
    for pool_id, army_count in placed_by_pool.items():
        if army_count > remaining_by_pool[pool_id]:
            raise InvalidActionError(
                f"Placement exceeds the remaining armies in pool {pool_id!r}."
            )
        if pool_definitions[pool_id].remaining_armies <= 0:
            raise InvalidActionError(f"Reinforcement pool {pool_id!r} is exhausted.")


def validate_action(
    state: FullGameState,
    action: Action,
    *,
    board: BoardLike = WAR1_BOARD_SPEC.map,
) -> None:
    """Validate an action without changing state or consuming randomness."""
    if state.is_finished:
        raise InvalidActionError("Finished games do not accept actions.")
    if action.expected_state_version != state.state_version:
        raise InvalidActionError(
            f"Action targets state version {action.expected_state_version}; "
            f"current version is {state.state_version}."
        )

    decision_player_id = _current_decision_player_id(state)
    if decision_player_id is None:
        raise InvalidActionError("The state has no player responsible for a decision.")
    if action.actor_player_id != decision_player_id:
        raise InvalidActionError(
            f"Player {action.actor_player_id!r} is not responsible for the next "
            f"decision; expected {decision_player_id!r}."
        )
    board_territory_ids = {territory.territory_id for territory in board.territories}
    if any(
        territory.territory_id not in board_territory_ids
        for territory in state.territories
    ):
        raise InvalidActionError("State contains a territory absent from BoardSpec.")

    operation = action.operation
    if isinstance(operation, ClaimTerritoryOperation):
        if state.phase_id != "territory_claim":
            raise InvalidActionError(
                f"Territory claims are not legal during {state.phase_id!r}."
            )
        territory = _territory_by_id(state).get(operation.territory_id)
        if territory is None:
            raise InvalidActionError(
                f"Unknown territory ID {operation.territory_id!r}."
            )
        if operation.territory_id not in board_territory_ids:
            raise InvalidActionError(
                f"Territory {operation.territory_id!r} is not in BoardSpec."
            )
        if territory.owner_player_id is not None:
            raise InvalidActionError(
                f"Territory {operation.territory_id!r} is already owned."
            )
        return
    if isinstance(operation, PlaceArmiesOperation):
        if (
            state.phase_id == "initial_placement"
            and action.actor_player_id not in state.initial_placement_order
        ):
            raise InvalidActionError(
                "Initial placement requires an explicit player order."
            )
        _validate_place_armies(state, action, operation)
        return

    raise UnsupportedPhaseError(
        f"{operation.action_type!r} transitions are not implemented in Phase 2."
    )


def legal_actions(
    state: FullGameState,
    *,
    board: BoardLike = WAR1_BOARD_SPEC.map,
) -> tuple[Action, ...]:
    """Return composable legal actions for the supported setup phases."""
    decision_player_id = _current_decision_player_id(state)
    if state.is_finished or decision_player_id is None:
        return ()
    if state.phase_id not in {
        "territory_claim",
        "initial_placement",
        "reinforcement_placement",
    }:
        raise UnsupportedPhaseError(
            f"Legal actions for phase {state.phase_id!r} are not implemented in "
            "Phase 2."
        )

    candidates: list[Action] = []
    if state.phase_id == "territory_claim":
        for territory in state.territories:
            if territory.owner_player_id is not None:
                continue
            action = Action(
                contract_type="action",
                action_id=f"legal-{state.state_version}-{len(candidates)}",
                actor_player_id=decision_player_id,
                action_schema_version=ACTION_SCHEMA_VERSION,
                expected_state_version=state.state_version,
                operation=ClaimTerritoryOperation(
                    action_type="claim_territory",
                    territory_id=territory.territory_id,
                ),
            )
            validate_action(state, action, board=board)
            candidates.append(action)
    else:
        placement_phase: Literal["initial", "reinforcement"] = (
            "initial" if state.phase_id == "initial_placement" else "reinforcement"
        )
        for pool in _pools_for_player(state, decision_player_id):
            for territory in state.territories:
                if territory.owner_player_id != decision_player_id:
                    continue
                if (
                    pool.allowed_territory_ids is not None
                    and territory.territory_id not in pool.allowed_territory_ids
                ):
                    continue
                placement = ArmyPlacement(
                    territory_id=territory.territory_id,
                    army_count=1,
                    source_pool_id=pool.pool_id,
                )
                action = Action(
                    contract_type="action",
                    action_id=f"legal-{state.state_version}-{len(candidates)}",
                    actor_player_id=decision_player_id,
                    action_schema_version=ACTION_SCHEMA_VERSION,
                    expected_state_version=state.state_version,
                    operation=PlaceArmiesOperation(
                        action_type="place_armies",
                        placements=(placement,),
                        placement_phase=placement_phase,
                    ),
                )
                validate_action(state, action, board=board)
                candidates.append(action)
    return tuple(candidates)


def _rebuild_state(
    state: FullGameState,
    *,
    territories: tuple[TerritoryState, ...],
    reinforcement_pools: tuple[ReinforcementPool, ...],
    phase_id: str | None = None,
    active_player_id: str | None = None,
    turn_player_id: str | None = None,
    decision_player_id: str | None = None,
) -> FullGameState:
    player_territory_counts = dict.fromkeys(
        (player.public.player_id for player in state.players), 0
    )
    player_army_counts = dict.fromkeys(
        (player.public.player_id for player in state.players), 0
    )
    for territory in territories:
        if territory.owner_player_id is None:
            continue
        player_territory_counts[territory.owner_player_id] += 1
        player_army_counts[territory.owner_player_id] += territory.army_count

    players: list[FullPlayerState] = []
    for player in state.players:
        public_data = player.public.model_dump()
        public_data["territory_count"] = player_territory_counts[
            player.public.player_id
        ]
        public_data["army_count"] = player_army_counts[player.public.player_id]
        public_data["is_eliminated"] = (
            player_territory_counts[player.public.player_id] == 0
        )
        players.append(
            FullPlayerState(
                public=PublicPlayerState.model_validate(public_data),
                private=player.private,
            )
        )

    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "state_version": state.state_version + 1,
            "phase_id": phase_id if phase_id is not None else state.phase_id,
            "active_player_id": (
                active_player_id
                if active_player_id is not None
                or phase_id == "initial_placement_complete"
                else state.active_player_id
            ),
            "turn_player_id": (
                turn_player_id
                if turn_player_id is not None
                or phase_id == "initial_placement_complete"
                else state.turn_player_id
            ),
            "decision_player_id": (
                decision_player_id
                if decision_player_id is not None
                or phase_id == "initial_placement_complete"
                else state.decision_player_id
            ),
            "players": tuple(players),
            "territories": territories,
            "reinforcement_pools": reinforcement_pools,
        }
    )
    return FullGameState.model_validate(state_data)


def _next_initial_placement_player(
    state: FullGameState,
    player_id: str,
    remaining_pools: tuple[ReinforcementPool, ...],
) -> str | None:
    order = state.initial_placement_order
    current_index = order.index(player_id)
    pending_player_ids = {pool.player_id for pool in remaining_pools}
    for offset in range(1, len(order) + 1):
        candidate = order[(current_index + offset) % len(order)]
        if candidate in pending_player_ids:
            return candidate
    return None


def apply_action(
    state: FullGameState,
    action: Action,
    *,
    board: BoardLike = WAR1_BOARD_SPEC.map,
) -> FullGameState:
    """Apply a validated setup action as a pure state transition."""
    validate_action(state, action, board=board)
    operation = action.operation
    if isinstance(operation, ClaimTerritoryOperation):
        territories = tuple(
            TerritoryState(
                territory_id=territory.territory_id,
                owner_player_id=(
                    action.actor_player_id
                    if territory.territory_id == operation.territory_id
                    else territory.owner_player_id
                ),
                army_count=(
                    1
                    if territory.territory_id == operation.territory_id
                    else territory.army_count
                ),
            )
            for territory in state.territories
        )
        order = state.initial_placement_order or tuple(
            player.public.player_id for player in state.players
        )
        if action.actor_player_id not in order:
            raise InvalidActionError(
                "Territory-claim order does not include the acting player."
            )
        current_index = order.index(action.actor_player_id)
        claim_next_player_id = order[(current_index + 1) % len(order)]
        claim_next_phase_id = (
            "territory_claim"
            if any(territory.owner_player_id is None for territory in territories)
            else "initial_placement"
        )
        return _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=state.reinforcement_pools,
            phase_id=claim_next_phase_id,
            active_player_id=claim_next_player_id,
            turn_player_id=claim_next_player_id,
            decision_player_id=claim_next_player_id,
        )
    if isinstance(operation, PlaceArmiesOperation):
        army_additions: dict[str, int] = {}
        pool_spend: dict[str, int] = {}
        for placement in operation.placements:
            pools = _eligible_pools_for_placement(
                state,
                action.actor_player_id,
                placement.territory_id,
                placement.source_pool_id,
            )
            pool = pools[0]
            army_additions[placement.territory_id] = (
                army_additions.get(placement.territory_id, 0) + placement.army_count
            )
            pool_spend[pool.pool_id] = (
                pool_spend.get(pool.pool_id, 0) + placement.army_count
            )

        territories = tuple(
            TerritoryState(
                territory_id=territory.territory_id,
                owner_player_id=territory.owner_player_id,
                army_count=territory.army_count
                + army_additions.get(territory.territory_id, 0),
            )
            for territory in state.territories
        )
        reinforcement_pools = tuple(
            ReinforcementPool.model_validate(
                {
                    **pool.model_dump(),
                    "remaining_armies": pool.remaining_armies
                    - pool_spend.get(pool.pool_id, 0),
                }
            )
            for pool in state.reinforcement_pools
            if pool.remaining_armies - pool_spend.get(pool.pool_id, 0) > 0
        )

        next_phase_id: str | None = None
        next_player_id: str | None = None
        if not any(
            pool.player_id == action.actor_player_id for pool in reinforcement_pools
        ):
            if state.phase_id == "initial_placement":
                next_player_id = _next_initial_placement_player(
                    state, action.actor_player_id, reinforcement_pools
                )
                if next_player_id is None:
                    next_phase_id = "initial_placement_complete"
            else:
                next_phase_id = "attack"
                next_player_id = action.actor_player_id

        return _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=reinforcement_pools,
            phase_id=next_phase_id,
            active_player_id=next_player_id,
            turn_player_id=next_player_id,
            decision_player_id=next_player_id,
        )
    raise UnsupportedPhaseError(
        f"{operation.action_type!r} transitions are not implemented in Phase 2."
    )


__all__ = [
    "InvalidActionError",
    "UnsupportedPhaseError",
    "apply_action",
    "legal_actions",
    "validate_action",
]
