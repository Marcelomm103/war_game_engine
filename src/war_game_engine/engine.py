"""Pure validation and transitions for the currently implemented game phases."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Literal, TypeAlias

from pydantic import ValidationError

from .board_spec import WAR1_BOARD_SPEC
from .combat import CombatDiceFaces, resolve_combat_round
from .contracts import (
    Action,
    AppliedChanceOutcome,
    ArmyPlacement,
    AttackOperation,
    BoardSpec,
    ChanceOutcome,
    ChanceRequest,
    ClaimTerritoryOperation,
    DeckState,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    EndPhaseOperation,
    EndTurnOperation,
    FullGameState,
    FullPlayerState,
    GamePhaseId,
    MapSpec,
    MoveArmiesOperation,
    MovementControl,
    MovementTransfer,
    ObjectiveResumePhaseId,
    OccupyTerritoryOperation,
    PendingCombat,
    PendingObjectiveCheck,
    PendingOccupation,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    RandomStreamsState,
    ReinforcementPool,
    TerritoryState,
    TradeCardsOperation,
)
from .objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    effective_objective_id,
    objective_winners,
)
from .randomness import RandomStreams
from .reinforcements import calculate_reinforcement_pools
from .versioning import ACTION_SCHEMA_VERSION

BoardLike: TypeAlias = MapSpec | BoardSpec


class InvalidActionError(ValueError):
    """Raised when an action is not legal for the current state."""


class UnsupportedPhaseError(ValueError):
    """Raised when an action's phase-specific rules are not implemented yet."""


class InvalidChanceOutcomeError(ValueError):
    """Raised when a combat outcome does not match its pending request."""


class VictoryAdjudicationRequiredError(ValueError):
    def __init__(
        self,
        state: FullGameState,
        winner_player_ids: tuple[str, ...],
    ) -> None:
        self.state = state
        self.winner_player_ids = winner_player_ids
        super().__init__(
            "Multiple players satisfy their objectives after one action: "
            f"{', '.join(winner_player_ids)}."
        )


def _map_spec(board: BoardLike) -> MapSpec:
    return board.map if isinstance(board, BoardSpec) else board


def _wildcard_card_ids(board: BoardLike) -> tuple[str, ...]:
    if isinstance(board, BoardSpec):
        return board.wildcard_card_ids
    return WAR1_BOARD_SPEC.wildcard_card_ids


def _known_card_ids(board: BoardLike) -> set[str]:
    return {
        territory.territory_id for territory in _map_spec(board).territories
    } | set(_wildcard_card_ids(board))


def _trade_set_is_valid(card_ids: tuple[str, ...], board: BoardLike) -> bool:
    if len(card_ids) != 3:
        return False
    map_spec = _map_spec(board)
    wildcard_ids = set(_wildcard_card_ids(board))
    territory_symbols = {
        territory.territory_id: territory.card_symbol
        for territory in map_spec.territories
    }
    unknown_card_ids = sorted(set(card_ids) - _known_card_ids(board))
    if unknown_card_ids:
        raise InvalidActionError(
            f"Unknown territory card IDs: {', '.join(unknown_card_ids)}."
        )

    fixed_symbols: list[str] = []
    for card_id in card_ids:
        if card_id in wildcard_ids:
            continue
        symbol = territory_symbols[card_id]
        if symbol is None:
            raise InvalidActionError(
                f"Territory card {card_id!r} has no certified symbol."
            )
        fixed_symbols.append(symbol)
    distinct_symbols = set(fixed_symbols)
    return len(distinct_symbols) == 1 or len(distinct_symbols) == len(fixed_symbols)


def _trade_army_bonus(trade_number: int) -> int:
    if trade_number <= 0:
        raise ValueError("Trade numbers are one-based and must be positive.")
    values = (4, 6, 8, 10, 12, 15, 20)
    if trade_number <= len(values):
        return values[trade_number - 1]
    return 20 + 5 * (trade_number - 7)


def _adjacent(map_spec: MapSpec, first_territory_id: str, second_territory_id: str) -> bool:
    return any(
        (
            adjacency.first_territory_id == first_territory_id
            and adjacency.second_territory_id == second_territory_id
        )
        or (
            adjacency.first_territory_id == second_territory_id
            and adjacency.second_territory_id == first_territory_id
        )
        for adjacency in map_spec.adjacencies
    )


def _current_decision_player_id(state: FullGameState) -> str | None:
    return state.decision_player_id or state.active_player_id


def _objective_resume_phase_id(
    phase_id: GamePhaseId,
) -> ObjectiveResumePhaseId:
    if phase_id == "objective_check" or phase_id == "finished":
        raise InvalidActionError(
            "Objective checks cannot resume into an objective-check or finished phase."
        )
    return phase_id


def _player_by_id(
    state: FullGameState, player_id: str
) -> FullPlayerState | None:
    return next(
        (player for player in state.players if player.public.player_id == player_id),
        None,
    )


def _players_with_hands(
    players: tuple[FullPlayerState, ...],
    hand_changes: dict[str, tuple[str, ...]],
) -> tuple[FullPlayerState, ...]:
    updated_players: list[FullPlayerState] = []
    for player in players:
        hand_card_ids = hand_changes.get(
            player.public.player_id, player.private.hand_card_ids
        )
        private_data = player.private.model_dump()
        private_data["hand_card_ids"] = hand_card_ids
        updated_players.append(
            FullPlayerState(
                public=player.public,
                private=PlayerPrivateState.model_validate(private_data),
            )
        )
    return tuple(updated_players)


def _players_after_elimination(
    players: tuple[FullPlayerState, ...],
    eliminated_player_id: str,
    eliminator_player_id: str,
) -> tuple[FullPlayerState, ...]:
    eliminated = next(
        (
            player
            for player in players
            if player.public.player_id == eliminated_player_id
        ),
        None,
    )
    if eliminated is None or eliminated_player_id == eliminator_player_id:
        raise InvalidActionError("Elimination references an invalid player.")
    updated_players: list[FullPlayerState] = []
    for player in players:
        public_data = player.public.model_dump()
        private_data = player.private.model_dump()
        if player.public.player_id == eliminated_player_id:
            public_data["eliminated_by_player_id"] = eliminator_player_id
        elif (
            player.public.player_id != eliminator_player_id
            and not player.public.is_eliminated
            and player.private.objective_id is not None
            and ELIMINATION_OBJECTIVE_COLORS.get(player.private.objective_id)
            == eliminated.public.color
        ):
            private_data["objective_fallback_id"] = FALLBACK_OBJECTIVE_ID
        updated_players.append(
            FullPlayerState(
                public=PublicPlayerState.model_validate(public_data),
                private=PlayerPrivateState.model_validate(private_data),
            )
        )
    return tuple(updated_players)


def _territory_by_id(state: FullGameState) -> dict[str, TerritoryState]:
    return {territory.territory_id: territory for territory in state.territories}


def _unmoved_army_count(state: FullGameState, territory_id: str) -> int:
    territory = _territory_by_id(state)[territory_id]
    armies_moved_into_territory = sum(
        transfer.army_count
        for transfer in state.movement_control.moved_transfers
        if transfer.target_territory_id == territory_id
    )
    return territory.army_count - armies_moved_into_territory


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
    if state.phase_id == "reinforcement_placement" and any(
        pool.player_id != action.actor_player_id for pool in state.reinforcement_pools
    ):
        raise InvalidActionError(
            "Reinforcement placement may only use the active player's pools."
        )

    territories = _territory_by_id(state)
    placement_territories = tuple(
        placement.territory_id for placement in operation.placements
    )
    # PlaceArmiesOperation is contract-limited to one placement.
    if len(placement_territories) != len(set(placement_territories)):  # pragma: no cover
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
        # Each operation places one troop and reinforcement pools are positive.
        if army_count > remaining_by_pool[pool_id]:  # pragma: no cover
            raise InvalidActionError(
                f"Placement exceeds the remaining armies in pool {pool_id!r}."
            )
        if pool_definitions[pool_id].remaining_armies <= 0:  # pragma: no cover
            raise InvalidActionError(f"Reinforcement pool {pool_id!r} is exhausted.")


def _validate_trade_cards(
    state: FullGameState,
    action: Action,
    operation: TradeCardsOperation,
    board: BoardLike,
) -> None:
    if state.phase_id != "reinforcement":
        raise InvalidActionError(
            f"'trade_cards' is not legal during {state.phase_id!r}."
        )
    player_pools = _pools_for_player(state, action.actor_player_id)
    if not player_pools or len(player_pools) != len(state.reinforcement_pools):
        raise InvalidActionError(
            "Reinforcement pools must be prepared for the active player before "
            "trading cards."
        )

    player = _player_by_id(state, action.actor_player_id)
    if player is None:
        raise InvalidActionError(
            f"Unknown player ID {action.actor_player_id!r}."
        )
    card_ids = operation.card_ids
    if len(set(card_ids)) != len(card_ids):
        raise InvalidActionError("A card trade must use three distinct cards.")
    hand_card_ids = set(player.private.hand_card_ids)
    cards_not_in_hand = sorted(set(card_ids) - hand_card_ids)
    if cards_not_in_hand:
        raise InvalidActionError(
            "Cards offered for trade are not in the player's hand: "
            f"{', '.join(cards_not_in_hand)}."
        )
    if not _trade_set_is_valid(card_ids, board):
        raise InvalidActionError(
            "A card trade must contain three equal or three different symbols, "
            "using wildcards as needed."
        )


def _validate_draw_card(state: FullGameState, board: BoardLike) -> None:
    if state.phase_id != "card_purchase":
        raise InvalidActionError(
            f"'draw_territory_card' is not legal during {state.phase_id!r}."
        )
    if not state.deck.draw_pile_card_ids and not state.deck.discard_pile_card_ids:
        raise InvalidActionError(
            "No territory cards are available in the draw or traded-card piles."
        )
    unknown_card_ids = sorted(
        (set(state.deck.draw_pile_card_ids) | set(state.deck.discard_pile_card_ids))
        - _known_card_ids(board)
    )
    if unknown_card_ids:
        raise InvalidActionError(
            f"Unknown territory card IDs in deck: {', '.join(unknown_card_ids)}."
        )
    if not state.deck.draw_pile_card_ids and state.random_streams_state is None:
        raise InvalidActionError(
            "Card-pile recycling requires the saved cards random stream."
        )


def _validate_discard_card(
    state: FullGameState,
    action: Action,
    operation: DiscardTerritoryCardOperation,
) -> None:
    if state.phase_id != "card_discard":
        raise InvalidActionError(
            f"'discard_territory_card' is not legal during {state.phase_id!r}."
        )
    player = _player_by_id(state, action.actor_player_id)
    if player is None or operation.card_id not in player.private.hand_card_ids:
        raise InvalidActionError(
            f"Card {operation.card_id!r} is not in the player's hand."
        )


def _validate_attack(
    state: FullGameState,
    action: Action,
    operation: AttackOperation,
    board: BoardLike,
) -> None:
    if state.phase_id != "attack":
        raise InvalidActionError(
            f"'attack' is not legal during {state.phase_id!r}."
        )
    if state.pending_combat is not None:
        raise InvalidActionError("A combat outcome is required before another action.")

    territories = _territory_by_id(state)
    origin = territories.get(operation.origin_territory_id)
    target = territories.get(operation.target_territory_id)
    if origin is None or target is None:
        raise InvalidActionError("Attack references an unknown territory.")
    if origin.owner_player_id != action.actor_player_id:
        raise InvalidActionError("Attack origin must belong to the acting player.")
    if origin.army_count < 2:
        raise InvalidActionError(
            "Attack origin must retain one army and have at least two armies."
        )
    if target.owner_player_id is None:
        raise InvalidActionError("An unowned territory cannot be attacked.")
    if target.owner_player_id == action.actor_player_id:
        raise InvalidActionError("Attack target must belong to an opponent.")
    if not _adjacent(
        _map_spec(board),
        operation.origin_territory_id,
        operation.target_territory_id,
    ):
        raise InvalidActionError("Attack territories must be directly adjacent.")
    if operation.dice_count > min(3, origin.army_count - 1):
        raise InvalidActionError(
            "Attack dice exceed the armies available after leaving the garrison."
        )


def _validate_occupation(
    state: FullGameState,
    action: Action,
    operation: OccupyTerritoryOperation,
) -> None:
    if state.phase_id != "occupation":
        raise InvalidActionError(
            f"'occupy_territory' is not legal during {state.phase_id!r}."
        )
    pending = state.pending_occupation
    if pending is None:
        raise InvalidActionError("Occupation requires a pending conquest.")
    if pending.attacker_player_id != action.actor_player_id:
        raise InvalidActionError("Only the attacker may occupy the conquered territory.")
    if (
        operation.origin_territory_id != pending.origin_territory_id
        or operation.target_territory_id != pending.target_territory_id
    ):
        raise InvalidActionError(
            "Occupation must use the origin and target of the pending conquest."
        )
    if operation.army_count > pending.maximum_army_count:
        raise InvalidActionError(
            "Occupation exceeds the armies available from the last attack."
        )


def _validate_movement(
    state: FullGameState,
    action: Action,
    operation: MoveArmiesOperation,
    board: BoardLike,
) -> None:
    if state.phase_id != "movement":
        raise InvalidActionError(
            f"'move_armies' is not legal during {state.phase_id!r}."
        )
    if operation.origin_territory_id == operation.target_territory_id:
        raise InvalidActionError("Movement cannot target its origin territory.")
    territories = _territory_by_id(state)
    origin = territories.get(operation.origin_territory_id)
    target = territories.get(operation.target_territory_id)
    if origin is None or target is None:
        raise InvalidActionError("Movement references an unknown territory.")
    if (
        origin.owner_player_id != action.actor_player_id
        or target.owner_player_id != action.actor_player_id
    ):
        raise InvalidActionError(
            "Movement origin and target must both belong to the acting player."
        )
    if not _adjacent(
        _map_spec(board),
        operation.origin_territory_id,
        operation.target_territory_id,
    ):
        raise InvalidActionError(
            "Movement territories must be directly adjacent; multi-step routes "
            "are not legal."
        )
    available_unmoved = _unmoved_army_count(state, operation.origin_territory_id)
    max_movable = min(available_unmoved, origin.army_count - 1)
    if operation.army_count > max_movable:
        raise InvalidActionError(
            "Movement exceeds unmoved armies or would leave no garrison."
        )


def validate_action(
    state: FullGameState,
    action: Action,
    *,
    board: BoardLike = WAR1_BOARD_SPEC,
) -> None:
    """Validate an action without changing state or consuming randomness."""
    if state.is_finished:
        raise InvalidActionError("Finished games do not accept actions.")
    if state.pending_combat is not None:
        raise InvalidActionError(
            "A combat outcome is required before another action."
        )
    if state.pending_objective_check is not None:
        raise InvalidActionError(
            "The objective check must be resolved before another action."
        )
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
    map_spec = _map_spec(board)
    board_territory_ids = {
        territory.territory_id for territory in map_spec.territories
    }
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
    if isinstance(operation, EndPhaseOperation):
        if state.phase_id != operation.phase_id:
            raise InvalidActionError(
                f"Cannot end phase {operation.phase_id!r} while the current phase is "
                f"{state.phase_id!r}."
            )
        if operation.phase_id == "reinforcement":
            player_pools = _pools_for_player(state, action.actor_player_id)
            if not player_pools or len(player_pools) != len(
                state.reinforcement_pools
            ):
                raise InvalidActionError(
                    "Reinforcement pools must be prepared for the active player "
                    "before leaving the reinforcement phase."
                )
        elif state.reinforcement_pools:
            raise InvalidActionError(
                "All reinforcement pools must be spent before ending this phase."
            )
        return
    if isinstance(operation, EndTurnOperation):
        if state.phase_id != "turn_end":
            raise InvalidActionError(
                f"Turns cannot end during {state.phase_id!r}."
            )
        if state.reinforcement_pools:
            raise InvalidActionError(
                "All reinforcement pools must be spent before the turn ends."
            )
        if (
            not state.initial_placement_order
            or action.actor_player_id not in state.initial_placement_order
        ):
            raise InvalidActionError(
                "Turn order must include the player ending the turn."
            )
        return

    if isinstance(operation, TradeCardsOperation):
        _validate_trade_cards(state, action, operation, board)
        return
    if isinstance(operation, DrawTerritoryCardOperation):
        _validate_draw_card(state, board)
        return
    if isinstance(operation, DiscardTerritoryCardOperation):
        _validate_discard_card(state, action, operation)
        return
    if isinstance(operation, AttackOperation):
        _validate_attack(state, action, operation, board)
        return
    if isinstance(operation, OccupyTerritoryOperation):
        _validate_occupation(state, action, operation)
        return
    if isinstance(operation, MoveArmiesOperation):
        _validate_movement(state, action, operation, board)
        return

    raise UnsupportedPhaseError(
        f"{operation.action_type!r} rules are not implemented yet."
    )


def legal_actions(
    state: FullGameState,
    *,
    board: BoardLike = WAR1_BOARD_SPEC,
) -> tuple[Action, ...]:
    """Return legal actions for setup and implemented game phases."""
    decision_player_id = _current_decision_player_id(state)
    if state.is_finished or decision_player_id is None:
        return ()
    if state.pending_combat is not None or state.pending_objective_check is not None:
        return ()
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
        return tuple(candidates)

    if state.phase_id in {"initial_placement", "reinforcement_placement"}:
        if state.phase_id == "reinforcement_placement" and not _pools_for_player(
            state, decision_player_id
        ):
            raise UnsupportedPhaseError(
            "Reinforcement pools are not prepared for this turn yet."
            )
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

    if state.phase_id == "reinforcement":
        if not _pools_for_player(state, decision_player_id):
            raise UnsupportedPhaseError(
                "Reinforcement pools are not prepared for this turn yet."
            )
        player = _player_by_id(state, decision_player_id)
        if player is None:
            raise InvalidActionError(
                f"Unknown player ID {decision_player_id!r}."
            )
        for card_ids in combinations(player.private.hand_card_ids, 3):
            if not _trade_set_is_valid(card_ids, board):
                continue
            action = Action(
                contract_type="action",
                action_id=f"legal-{state.state_version}-{len(candidates)}",
                actor_player_id=decision_player_id,
                action_schema_version=ACTION_SCHEMA_VERSION,
                expected_state_version=state.state_version,
                operation=TradeCardsOperation(
                    action_type="trade_cards",
                    card_ids=card_ids,
                ),
            )
            validate_action(state, action, board=board)
            candidates.append(action)

        action = Action(
            contract_type="action",
            action_id=f"legal-{state.state_version}-{len(candidates)}",
            actor_player_id=decision_player_id,
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=EndPhaseOperation(
                action_type="end_phase",
                phase_id="reinforcement",
            ),
        )
        validate_action(state, action, board=board)
        candidates.append(action)
        return tuple(candidates)

    if state.phase_id == "attack":
        map_spec = _map_spec(board)
        territories = _territory_by_id(state)
        for origin in state.territories:
            if (
                origin.owner_player_id != decision_player_id
                or origin.army_count < 2
            ):
                continue
            max_dice_count = min(3, origin.army_count - 1)
            for adjacency in map_spec.adjacencies:
                if adjacency.first_territory_id == origin.territory_id:
                    target_id = adjacency.second_territory_id
                elif adjacency.second_territory_id == origin.territory_id:
                    target_id = adjacency.first_territory_id
                else:
                    continue
                target = territories.get(target_id)
                if (
                    target is None
                    or target.owner_player_id is None
                    or target.owner_player_id == decision_player_id
                ):
                    continue
                for dice_count in range(1, max_dice_count + 1):
                    action = Action(
                        contract_type="action",
                        action_id=f"legal-{state.state_version}-{len(candidates)}",
                        actor_player_id=decision_player_id,
                        action_schema_version=ACTION_SCHEMA_VERSION,
                        expected_state_version=state.state_version,
                        operation=AttackOperation(
                            action_type="attack",
                            origin_territory_id=origin.territory_id,
                            target_territory_id=target_id,
                            dice_count=dice_count,
                        ),
                    )
                    validate_action(state, action, board=board)
                    candidates.append(action)
        end_attack_action = Action(
            contract_type="action",
            action_id=f"legal-{state.state_version}-{len(candidates)}",
            actor_player_id=decision_player_id,
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=EndPhaseOperation(
                action_type="end_phase",
                phase_id="attack",
            ),
        )
        validate_action(state, end_attack_action, board=board)
        candidates.append(end_attack_action)
        return tuple(candidates)

    if state.phase_id == "movement":
        map_spec = _map_spec(board)
        territories = _territory_by_id(state)
        for origin in state.territories:
            if origin.owner_player_id != decision_player_id:
                continue
            if min(
                _unmoved_army_count(state, origin.territory_id),
                origin.army_count - 1,
            ) < 1:
                continue
            for adjacency in map_spec.adjacencies:
                if adjacency.first_territory_id == origin.territory_id:
                    target_id = adjacency.second_territory_id
                elif adjacency.second_territory_id == origin.territory_id:
                    target_id = adjacency.first_territory_id
                else:
                    continue
                target = territories.get(target_id)
                if (
                    target is None
                    or target.owner_player_id != decision_player_id
                ):
                    continue
                action = Action(
                    contract_type="action",
                    action_id=f"legal-{state.state_version}-{len(candidates)}",
                    actor_player_id=decision_player_id,
                    action_schema_version=ACTION_SCHEMA_VERSION,
                    expected_state_version=state.state_version,
                    operation=MoveArmiesOperation(
                        action_type="move_armies",
                        origin_territory_id=origin.territory_id,
                        target_territory_id=target_id,
                        army_count=1,
                    ),
                )
                validate_action(state, action, board=board)
                candidates.append(action)

        action = Action(
            contract_type="action",
            action_id=f"legal-{state.state_version}-{len(candidates)}",
            actor_player_id=decision_player_id,
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=EndPhaseOperation(
                action_type="end_phase",
                phase_id="movement",
            ),
        )
        validate_action(state, action, board=board)
        candidates.append(action)
        return tuple(candidates)

    if state.phase_id == "card_purchase":
        action = Action(
            contract_type="action",
            action_id=f"legal-{state.state_version}-0",
            actor_player_id=decision_player_id,
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=DrawTerritoryCardOperation(
                action_type="draw_territory_card"
            ),
        )
        validate_action(state, action, board=board)
        return (action,)

    if state.phase_id == "card_discard":
        player = _player_by_id(state, decision_player_id)
        if player is None:
            raise InvalidActionError(
                f"Unknown player ID {decision_player_id!r}."
            )
        for card_id in player.private.hand_card_ids:
            action = Action(
                contract_type="action",
                action_id=f"legal-{state.state_version}-{len(candidates)}",
                actor_player_id=decision_player_id,
                action_schema_version=ACTION_SCHEMA_VERSION,
                expected_state_version=state.state_version,
                operation=DiscardTerritoryCardOperation(
                    action_type="discard_territory_card",
                    card_id=card_id,
                ),
            )
            validate_action(state, action, board=board)
            candidates.append(action)
        return tuple(candidates)

    if state.phase_id == "occupation":
        pending = state.pending_occupation
        if pending is None:
            raise InvalidActionError("Occupation requires a pending conquest.")
        for army_count in range(1, pending.maximum_army_count + 1):
            action = Action(
                contract_type="action",
                action_id=f"legal-{state.state_version}-{len(candidates)}",
                actor_player_id=decision_player_id,
                action_schema_version=ACTION_SCHEMA_VERSION,
                expected_state_version=state.state_version,
                operation=OccupyTerritoryOperation(
                    action_type="occupy_territory",
                    origin_territory_id=pending.origin_territory_id,
                    target_territory_id=pending.target_territory_id,
                    army_count=army_count,
                ),
            )
            validate_action(state, action, board=board)
            candidates.append(action)
        return tuple(candidates)

    if state.phase_id == "turn_end":
        action = Action(
            contract_type="action",
            action_id=f"legal-{state.state_version}-0",
            actor_player_id=decision_player_id,
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=EndTurnOperation(action_type="end_turn"),
        )
        validate_action(state, action, board=board)
        return (action,)

    raise UnsupportedPhaseError(
        f"Legal actions for phase {state.phase_id!r} are not implemented yet."
    )


def _rebuild_state(
    state: FullGameState,
    *,
    territories: tuple[TerritoryState, ...],
    reinforcement_pools: tuple[ReinforcementPool, ...],
    players: tuple[FullPlayerState, ...] | None = None,
    deck: DeckState | None = None,
    random_streams_state: RandomStreamsState | None = None,
    pending_combat: PendingCombat | None = None,
    pending_occupation: PendingOccupation | None = None,
    pending_objective_check: PendingObjectiveCheck | None = None,
    applied_chance_outcomes: tuple[AppliedChanceOutcome, ...] | None = None,
    winner_player_id: str | None = None,
    is_finished: bool | None = None,
    advance_state_version: bool = True,
    phase_id: GamePhaseId | None = None,
    active_player_id: str | None = None,
    turn_player_id: str | None = None,
    decision_player_id: str | None = None,
    turn_number: int | None = None,
    has_conquered_territory_this_turn: bool | None = None,
    movement_control: MovementControl | None = None,
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

    source_players = state.players if players is None else players
    updated_players: list[FullPlayerState] = []
    for player in source_players:
        public_data = player.public.model_dump()
        public_data["territory_count"] = player_territory_counts[
            player.public.player_id
        ]
        public_data["army_count"] = player_army_counts[player.public.player_id]
        public_data["is_eliminated"] = (
            player_territory_counts[player.public.player_id] == 0
        )
        updated_players.append(
            FullPlayerState(
                public=PublicPlayerState.model_validate(public_data),
                private=player.private,
            )
        )

    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "state_version": state.state_version + (1 if advance_state_version else 0),
            "phase_id": phase_id if phase_id is not None else state.phase_id,
            "turn_number": (
                turn_number if turn_number is not None else state.turn_number
            ),
            "has_conquered_territory_this_turn": (
                has_conquered_territory_this_turn
                if has_conquered_territory_this_turn is not None
                else state.has_conquered_territory_this_turn
            ),
            "active_player_id": (
                active_player_id
                if active_player_id is not None
                else state.active_player_id
            ),
            "turn_player_id": (
                turn_player_id if turn_player_id is not None else state.turn_player_id
            ),
            "decision_player_id": (
                decision_player_id
                if decision_player_id is not None
                else state.decision_player_id
            ),
            "players": tuple(updated_players),
            "territories": territories,
            "reinforcement_pools": reinforcement_pools,
            "deck": deck if deck is not None else state.deck,
            "random_streams_state": (
                random_streams_state
                if random_streams_state is not None
                else state.random_streams_state
            ),
            "pending_combat": pending_combat,
            "pending_occupation": pending_occupation,
            "pending_objective_check": pending_objective_check,
            "applied_chance_outcomes": (
                applied_chance_outcomes
                if applied_chance_outcomes is not None
                else state.applied_chance_outcomes
            ),
            "winner_player_id": winner_player_id,
            "is_finished": is_finished if is_finished is not None else state.is_finished,
            "movement_control": (
                movement_control
                if movement_control is not None
                else state.movement_control
            ),
        }
    )
    if is_finished:
        state_data.update(
            {
                "active_player_id": None,
                "turn_player_id": None,
                "decision_player_id": None,
            }
        )
    return FullGameState.model_validate(state_data)


def _transfer_eliminated_cards(
    state: FullGameState,
    recipient_player_id: str,
    eliminated_player_id: str,
) -> tuple[tuple[FullPlayerState, ...], DeckState, RandomStreamsState | None]:
    """Transfer an eliminated hand when that elimination does not end the game."""
    if recipient_player_id == eliminated_player_id:
        raise InvalidActionError("A player cannot receive their own eliminated hand.")
    recipient = _player_by_id(state, recipient_player_id)
    eliminated = _player_by_id(state, eliminated_player_id)
    if recipient is None or recipient.public.is_eliminated:
        raise InvalidActionError(
            f"Card recipient {recipient_player_id!r} is not an active player."
        )
    if eliminated is None or not eliminated.public.is_eliminated:
        raise InvalidActionError(
            f"Player {eliminated_player_id!r} has not been eliminated."
        )

    combined_hand = (
        *recipient.private.hand_card_ids,
        *eliminated.private.hand_card_ids,
    )
    discarded_card_ids: tuple[str, ...] = ()
    random_streams_state = state.random_streams_state
    excess_cards = max(0, len(combined_hand) - 5)
    if excess_cards:
        if random_streams_state is None:
            raise InvalidActionError(
                "Random card discard after elimination requires the saved cards "
                "random stream."
            )
        streams = RandomStreams.restore(random_streams_state)
        discarded_card_ids = streams.sample(
            "cards",
            combined_hand,
            excess_cards,
        )
        random_streams_state = streams.snapshot()

    discarded_set = set(discarded_card_ids)
    recipient_hand = tuple(
        card_id for card_id in combined_hand if card_id not in discarded_set
    )
    players = _players_with_hands(
        state.players,
        {
            recipient_player_id: recipient_hand,
            eliminated_player_id: (),
        },
    )
    deck = DeckState.model_validate(
        {
            **state.deck.model_dump(),
            "hand_limit_discard_card_ids": (
                *state.deck.hand_limit_discard_card_ids,
                *discarded_card_ids,
            ),
        }
    )
    return players, deck, random_streams_state


def resolve_pending_objective_check(
    state: FullGameState,
    *,
    board: BoardLike = WAR1_BOARD_SPEC,
) -> FullGameState:
    pending = state.pending_objective_check
    if state.phase_id != "objective_check" or pending is None:
        raise InvalidActionError("The state has no pending objective check.")

    winner_player_ids = objective_winners(state, board=board)
    if len(winner_player_ids) > 1:
        raise VictoryAdjudicationRequiredError(state, winner_player_ids)
    winner_player_id = winner_player_ids[0] if winner_player_ids else None

    players = state.players
    deck = state.deck
    random_streams_state = state.random_streams_state
    if pending.eliminated_player_id is not None:
        eliminated = _player_by_id(state, pending.eliminated_player_id)
        attacker = _player_by_id(state, pending.actor_player_id)
        if eliminated is None or attacker is None:
            raise InvalidActionError(
                "Pending objective check references an unknown player."
            )
        attacker_has_elimination_objective = (
            ELIMINATION_OBJECTIVE_COLORS.get(
                effective_objective_id(attacker)
            )
            == eliminated.public.color
        )
        if not (
            winner_player_id == pending.actor_player_id
            and attacker_has_elimination_objective
        ):
            players, deck, random_streams_state = _transfer_eliminated_cards(
                state,
                pending.actor_player_id,
                pending.eliminated_player_id,
            )

    if winner_player_id is not None:
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=(),
            players=players,
            deck=deck,
            random_streams_state=random_streams_state,
            pending_combat=None,
            pending_occupation=None,
            pending_objective_check=None,
            winner_player_id=winner_player_id,
            is_finished=True,
            phase_id="finished",
            advance_state_version=False,
        )

    resume_phase_id = pending.resume_phase_id
    pending_occupation = (
        state.pending_occupation if resume_phase_id == "occupation" else None
    )
    return _rebuild_state(
        state,
        territories=state.territories,
        reinforcement_pools=state.reinforcement_pools,
        players=players,
        deck=deck,
        random_streams_state=random_streams_state,
        pending_combat=None,
        pending_occupation=pending_occupation,
        pending_objective_check=None,
        phase_id=resume_phase_id,
        active_player_id=pending.resume_decision_player_id,
        turn_player_id=pending.resume_decision_player_id,
        decision_player_id=pending.resume_decision_player_id,
        advance_state_version=False,
    )


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


def _first_active_player(state: FullGameState) -> str:
    active_player_ids = {
        player.public.player_id
        for player in state.players
        if not player.public.is_eliminated
    }
    for player_id in state.initial_placement_order:
        if player_id in active_player_ids:
            return player_id
    raise InvalidActionError("Initial placement order has no active player.")


def _next_turn_player(
    state: FullGameState,
    player_id: str,
) -> tuple[str, int]:
    order = state.initial_placement_order
    if not order or player_id not in order:
        raise InvalidActionError(
            "Turn order must include the player ending the turn."
        )

    current_index = order.index(player_id)
    active_player_ids = {
        player.public.player_id
        for player in state.players
        if not player.public.is_eliminated
    }
    for offset in range(1, len(order) + 1):
        candidate_index = (current_index + offset) % len(order)
        candidate_id = order[candidate_index]
        if candidate_id in active_player_ids:
            turn_number = state.turn_number + (
                1 if candidate_index <= current_index else 0
            )
            return candidate_id, turn_number
    raise InvalidActionError("Cannot advance a turn when no players are active.")


def chance_request_for_pending_combat(state: FullGameState) -> ChanceRequest:
    pending = state.pending_combat
    if pending is None:
        raise InvalidActionError("The state has no pending combat chance request.")
    return ChanceRequest(
        contract_type="chance_request",
        request_id=pending.request_id,
        game_id=state.game_id,
        request_type="combat_dice",
        requested_by_player_id=pending.attacker_player_id,
        parameters={
            "action_id": pending.action_id,
            "origin_territory_id": pending.origin_territory_id,
            "target_territory_id": pending.target_territory_id,
            "attacker_dice_count": pending.attacker_dice_count,
            "defender_dice_count": pending.defender_dice_count,
            "dice_sides": 6,
        },
    )


def simulate_pending_combat_outcome(
    state: FullGameState,
) -> tuple[ChanceOutcome, RandomStreamsState]:
    request = chance_request_for_pending_combat(state)
    pending = state.pending_combat
    if pending is None:
        raise InvalidActionError("The state has no pending combat chance request.")
    if state.random_streams_state is None:
        raise InvalidActionError(
            "Simulated combat requires the saved combat random stream."
        )

    streams = RandomStreams.restore(state.random_streams_state)
    faces = CombatDiceFaces(
        attacker_faces=tuple(
            streams.die("combat") for _ in range(pending.attacker_dice_count)
        ),
        defender_faces=tuple(
            streams.die("combat") for _ in range(pending.defender_dice_count)
        ),
    )
    return (
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=request.request_id,
            event_id=f"{request.request_id}:outcome",
            outcome=faces.model_dump(mode="json"),
            visibility="public",
            outcome_source="simulation",
        ),
        streams.snapshot(),
    )


def apply_combat_outcome(
    state: FullGameState,
    outcome: ChanceOutcome,
    *,
    random_streams_state: RandomStreamsState | None = None,
    board: BoardLike = WAR1_BOARD_SPEC,
) -> FullGameState:
    if not outcome.request_id.strip():
        raise InvalidChanceOutcomeError("Combat outcome request ID cannot be blank.")
    if not outcome.event_id.strip():
        raise InvalidChanceOutcomeError("Combat outcome event ID cannot be blank.")
    if any(
        applied.request_id == outcome.request_id
        for applied in state.applied_chance_outcomes
    ):
        raise InvalidChanceOutcomeError(
            "Combat chance request has already been applied."
        )
    if any(
        applied.event_id == outcome.event_id
        for applied in state.applied_chance_outcomes
    ):
        raise InvalidChanceOutcomeError("Chance outcome event ID has already been applied.")
    if outcome.outcome_source == "simulation":
        if state.random_streams_state is None or random_streams_state is None:
            raise InvalidChanceOutcomeError(
                "Simulated combat outcomes must include the updated random stream."
            )
    elif random_streams_state is not None:
        raise InvalidChanceOutcomeError(
            "Observed combat outcomes cannot advance internal random streams."
        )
    pending = state.pending_combat
    if pending is None:
        raise InvalidChanceOutcomeError("No combat is waiting for an outcome.")
    if outcome.request_id != pending.request_id:
        raise InvalidChanceOutcomeError(
            "Combat outcome request ID does not match the pending request."
        )
    if outcome.visibility != "public":
        raise InvalidChanceOutcomeError("Combat dice outcomes must be public.")
    try:
        faces = CombatDiceFaces.model_validate_json(json.dumps(outcome.outcome))
    except ValidationError as error:
        raise InvalidChanceOutcomeError(
            "Combat outcome must contain one to three six-sided faces for each side."
        ) from error
    if (
        len(faces.attacker_faces) != pending.attacker_dice_count
        or len(faces.defender_faces) != pending.defender_dice_count
    ):
        raise InvalidChanceOutcomeError(
            "Combat outcome face counts do not match the pending request."
        )

    territory_by_id = _territory_by_id(state)
    origin = territory_by_id[pending.origin_territory_id]
    target = territory_by_id[pending.target_territory_id]
    defender_player_id = target.owner_player_id
    if origin.owner_player_id != pending.attacker_player_id:
        raise InvalidChanceOutcomeError(
            "Territory ownership changed while combat was awaiting its outcome."
        )
    if defender_player_id is None or defender_player_id == pending.attacker_player_id:
        raise InvalidChanceOutcomeError(
            "Territory ownership changed while combat was awaiting its outcome."
        )

    attacker_losses, defender_losses = resolve_combat_round(
        faces.attacker_faces,
        faces.defender_faces,
    )
    origin_armies = origin.army_count - attacker_losses
    target_armies = target.army_count - defender_losses
    if origin_armies < 1 or target_armies < 0:
        raise InvalidChanceOutcomeError("Combat losses exceed available armies.")

    pending_occupation: PendingOccupation | None = None
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                None
                if (
                    territory.territory_id == pending.target_territory_id
                    and target_armies == 0
                )
                else territory.owner_player_id
            ),
            army_count=(
                origin_armies
                if territory.territory_id == pending.origin_territory_id
                else (
                    target_armies
                    if territory.territory_id == pending.target_territory_id
                    else territory.army_count
                )
            ),
        )
        for territory in state.territories
    )
    if target_armies == 0:
        maximum_army_count = min(
            pending.attacker_dice_count,
            origin_armies - 1,
        )
        if maximum_army_count < 1:
            raise InvalidChanceOutcomeError(
                "A conquered territory requires at least one available army."
            )
        pending_occupation = PendingOccupation(
            contract_type="pending_occupation",
            attacker_player_id=pending.attacker_player_id,
            defender_player_id=defender_player_id,
            origin_territory_id=pending.origin_territory_id,
            target_territory_id=pending.target_territory_id,
            maximum_army_count=maximum_army_count,
        )
    defender_eliminated = target_armies == 0 and not any(
        territory.owner_player_id == defender_player_id
        for territory in territories
    )
    players = (
        _players_after_elimination(
            state.players,
            defender_player_id,
            pending.attacker_player_id,
        )
        if defender_eliminated
        else state.players
    )
    pending_objective_check = PendingObjectiveCheck(
        contract_type="pending_objective_check",
        actor_player_id=pending.attacker_player_id,
        territory_id=pending.target_territory_id,
        resume_phase_id="occupation" if pending_occupation is not None else "attack",
        resume_decision_player_id=pending.attacker_player_id,
        eliminated_player_id=(
            defender_player_id if defender_eliminated else None
        ),
    )
    applied_chance_outcomes = (
        *state.applied_chance_outcomes,
        AppliedChanceOutcome(
            request_id=outcome.request_id,
            event_id=outcome.event_id,
            visibility=outcome.visibility,
            outcome_source=outcome.outcome_source,
        ),
    )
    pending_state = _rebuild_state(
        state,
        territories=territories,
        reinforcement_pools=state.reinforcement_pools,
        players=players,
        pending_combat=None,
        pending_occupation=pending_occupation,
        pending_objective_check=pending_objective_check,
        applied_chance_outcomes=applied_chance_outcomes,
        phase_id="objective_check",
        active_player_id=pending.attacker_player_id,
        turn_player_id=pending.attacker_player_id,
        decision_player_id=pending.attacker_player_id,
        random_streams_state=random_streams_state,
    )
    return resolve_pending_objective_check(pending_state, board=board)


def apply_action(
    state: FullGameState,
    action: Action,
    *,
    board: BoardLike = WAR1_BOARD_SPEC,
) -> FullGameState:
    """Apply a validated action as a pure state transition."""
    validate_action(state, action, board=board)
    operation = action.operation
    if isinstance(operation, EndPhaseOperation):
        if operation.phase_id == "reinforcement":
            end_phase_id: GamePhaseId = "reinforcement_placement"
        elif operation.phase_id == "attack":
            end_phase_id = "movement"
        else:
            end_phase_id = (
                "card_purchase"
                if state.has_conquered_territory_this_turn
                else "turn_end"
            )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=state.reinforcement_pools,
            phase_id=end_phase_id,
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
        )
    if isinstance(operation, EndTurnOperation):
        next_turn_player_id, next_round_number = _next_turn_player(
            state, action.actor_player_id
        )
        reinforcement_pools = calculate_reinforcement_pools(
            next_turn_player_id,
            state.territories,
            _map_spec(board),
            pool_scope=f"turn-{next_round_number}",
        )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=reinforcement_pools,
            phase_id="reinforcement",
            active_player_id=next_turn_player_id,
            turn_player_id=next_turn_player_id,
            decision_player_id=next_turn_player_id,
            turn_number=next_round_number,
            has_conquered_territory_this_turn=False,
            movement_control=MovementControl(),
        )
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
        claim_next_phase_id: GamePhaseId = (
            "territory_claim"
            if any(territory.owner_player_id is None for territory in territories)
            else "initial_placement"
        )
        pending_objective_check = PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=action.actor_player_id,
            territory_id=operation.territory_id,
            resume_phase_id=_objective_resume_phase_id(claim_next_phase_id),
            resume_decision_player_id=claim_next_player_id,
        )
        pending_state = _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=state.reinforcement_pools,
            pending_objective_check=pending_objective_check,
            phase_id="objective_check",
            active_player_id=claim_next_player_id,
            turn_player_id=claim_next_player_id,
            decision_player_id=claim_next_player_id,
        )
        return resolve_pending_objective_check(pending_state, board=board)
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

        next_phase_id: GamePhaseId | None = None
        next_player_id: str | None = None
        next_turn_number: int | None = None
        if not any(
            pool.player_id == action.actor_player_id for pool in reinforcement_pools
        ):
            if state.phase_id == "initial_placement":
                next_player_id = _next_initial_placement_player(
                    state, action.actor_player_id, reinforcement_pools
                )
                if next_player_id is None:
                    next_phase_id = "reinforcement"
                    next_player_id = _first_active_player(state)
                    next_turn_number = state.turn_number + 1
            else:
                next_phase_id = "attack"
                next_player_id = action.actor_player_id

        if next_phase_id == "reinforcement" and next_player_id is not None:
            reinforcement_pools = calculate_reinforcement_pools(
                next_player_id,
                territories,
                _map_spec(board),
                pool_scope=f"turn-{next_turn_number}",
            )

        resume_phase_id = _objective_resume_phase_id(
            next_phase_id or state.phase_id
        )
        resume_player_id = next_player_id or action.actor_player_id
        pending_objective_check = PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=action.actor_player_id,
            territory_id=operation.placements[0].territory_id,
            resume_phase_id=resume_phase_id,
            resume_decision_player_id=resume_player_id,
        )
        pending_state = _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=reinforcement_pools,
            pending_objective_check=pending_objective_check,
            phase_id="objective_check",
            active_player_id=resume_player_id,
            turn_player_id=resume_player_id,
            decision_player_id=resume_player_id,
            turn_number=(
                next_turn_number if next_turn_number is not None else state.turn_number
            ),
        )
        return resolve_pending_objective_check(pending_state, board=board)
    if isinstance(operation, AttackOperation):
        territory_by_id = _territory_by_id(state)
        target = territory_by_id[operation.target_territory_id]
        pending_combat = PendingCombat(
            contract_type="pending_combat",
            request_id=(
                f"{state.game_id}:combat:{state.state_version}:{action.action_id}"
            ),
            action_id=action.action_id,
            attacker_player_id=action.actor_player_id,
            origin_territory_id=operation.origin_territory_id,
            target_territory_id=operation.target_territory_id,
            attacker_dice_count=operation.dice_count,
            defender_dice_count=min(target.army_count, 3),
        )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=state.reinforcement_pools,
            pending_combat=pending_combat,
            phase_id="attack",
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
        )
    if isinstance(operation, OccupyTerritoryOperation):
        pending = state.pending_occupation
        if pending is None:
            raise InvalidActionError("Occupation requires a pending conquest.")
        territories = tuple(
            TerritoryState(
                territory_id=territory.territory_id,
                owner_player_id=(
                    action.actor_player_id
                    if territory.territory_id == pending.target_territory_id
                    else territory.owner_player_id
                ),
                army_count=(
                    territory.army_count - operation.army_count
                    if territory.territory_id == pending.origin_territory_id
                    else (
                        operation.army_count
                        if territory.territory_id == pending.target_territory_id
                        else territory.army_count
                    )
                ),
            )
            for territory in state.territories
        )
        pending_objective_check = PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=action.actor_player_id,
            territory_id=pending.target_territory_id,
            resume_phase_id="attack",
            resume_decision_player_id=action.actor_player_id,
        )
        pending_state = _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=state.reinforcement_pools,
            pending_combat=None,
            pending_occupation=None,
            pending_objective_check=pending_objective_check,
            phase_id="objective_check",
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
            has_conquered_territory_this_turn=True,
        )
        return resolve_pending_objective_check(pending_state, board=board)
    if isinstance(operation, MoveArmiesOperation):
        territories = tuple(
            TerritoryState(
                territory_id=territory.territory_id,
                owner_player_id=territory.owner_player_id,
                army_count=(
                    territory.army_count - operation.army_count
                    if territory.territory_id == operation.origin_territory_id
                    else (
                        territory.army_count + operation.army_count
                        if territory.territory_id == operation.target_territory_id
                        else territory.army_count
                    )
                ),
            )
            for territory in state.territories
        )
        route = (
            operation.origin_territory_id,
            operation.target_territory_id,
        )
        moved_transfers: list[MovementTransfer] = []
        route_updated = False
        for transfer in state.movement_control.moved_transfers:
            if (
                transfer.origin_territory_id,
                transfer.target_territory_id,
            ) == route:
                moved_transfers.append(
                    MovementTransfer(
                        origin_territory_id=transfer.origin_territory_id,
                        target_territory_id=transfer.target_territory_id,
                        army_count=transfer.army_count + operation.army_count,
                    )
                )
                route_updated = True
            else:
                moved_transfers.append(transfer)
        if not route_updated:
            moved_transfers.append(
                MovementTransfer(
                    origin_territory_id=operation.origin_territory_id,
                    target_territory_id=operation.target_territory_id,
                    army_count=operation.army_count,
                )
            )
        movement_control = MovementControl(
            moved_transfers=tuple(moved_transfers)
        )
        pending_objective_check = PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=action.actor_player_id,
            territory_id=operation.target_territory_id,
            resume_phase_id="movement",
            resume_decision_player_id=action.actor_player_id,
        )
        pending_state = _rebuild_state(
            state,
            territories=territories,
            reinforcement_pools=state.reinforcement_pools,
            pending_combat=None,
            pending_occupation=None,
            pending_objective_check=pending_objective_check,
            movement_control=movement_control,
            phase_id="objective_check",
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
        )
        return resolve_pending_objective_check(pending_state, board=board)
    if isinstance(operation, TradeCardsOperation):
        trade_player = _player_by_id(state, action.actor_player_id)
        if trade_player is None:
            raise InvalidActionError(
                f"Unknown player ID {action.actor_player_id!r}."
            )
        trade_number = state.deck.trade_count + 1
        trade_scope = (
            f"{action.actor_player_id}:turn-{state.turn_number}:trade-{trade_number}"
        )
        territory_by_id = _territory_by_id(state)
        trade_pools = [
            ReinforcementPool(
                pool_id=f"{trade_scope}:base",
                player_id=action.actor_player_id,
                origin="card_trade",
                remaining_armies=_trade_army_bonus(trade_number),
            )
        ]
        for card_id in operation.card_ids:
            if card_id in _wildcard_card_ids(board):
                continue
            territory = territory_by_id.get(card_id)
            if territory is None:
                raise InvalidActionError(
                    f"Game state is missing territory {card_id!r} represented by "
                    "a traded card."
                )
            if territory.owner_player_id == action.actor_player_id:
                trade_pools.append(
                    ReinforcementPool(
                        pool_id=f"{trade_scope}:territory-card:{card_id}",
                        player_id=action.actor_player_id,
                        origin="territory_card",
                        remaining_armies=2,
                        allowed_territory_ids=(card_id,),
                        source_card_id=card_id,
                    )
                )

        traded_card_ids = set(operation.card_ids)
        players = _players_with_hands(
            state.players,
            {
                action.actor_player_id: tuple(
                    card_id
                    for card_id in trade_player.private.hand_card_ids
                    if card_id not in traded_card_ids
                )
            },
        )
        deck = DeckState.model_validate(
            {
                **state.deck.model_dump(),
                "discard_pile_card_ids": (
                    *state.deck.discard_pile_card_ids,
                    *operation.card_ids,
                ),
                "trade_count": trade_number,
            }
        )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=(*state.reinforcement_pools, *trade_pools),
            players=players,
            deck=deck,
        )
    if isinstance(operation, DrawTerritoryCardOperation):
        random_streams_state: RandomStreamsState | None = None
        if state.deck.draw_pile_card_ids:
            card_id = state.deck.draw_pile_card_ids[0]
            draw_pile_card_ids = state.deck.draw_pile_card_ids[1:]
            discard_pile_card_ids = state.deck.discard_pile_card_ids
        else:
            if state.random_streams_state is None:
                raise InvalidActionError(
                    "Card-pile recycling requires the saved cards random stream."
                )
            streams = RandomStreams.restore(state.random_streams_state)
            shuffled_cards = streams.shuffle(
                "cards", state.deck.discard_pile_card_ids
            )
            card_id = shuffled_cards[0]
            draw_pile_card_ids = shuffled_cards[1:]
            discard_pile_card_ids = ()
            random_streams_state = streams.snapshot()

        draw_player = _player_by_id(state, action.actor_player_id)
        if draw_player is None:
            raise InvalidActionError(
                f"Unknown player ID {action.actor_player_id!r}."
            )
        hand_card_ids = (*draw_player.private.hand_card_ids, card_id)
        players = _players_with_hands(
            state.players,
            {action.actor_player_id: hand_card_ids},
        )
        deck = DeckState.model_validate(
            {
                **state.deck.model_dump(),
                "draw_pile_card_ids": draw_pile_card_ids,
                "discard_pile_card_ids": discard_pile_card_ids,
            }
        )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=state.reinforcement_pools,
            players=players,
            deck=deck,
            random_streams_state=random_streams_state,
            phase_id="card_discard" if len(hand_card_ids) > 5 else "turn_end",
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
        )
    if isinstance(operation, DiscardTerritoryCardOperation):
        discard_player = _player_by_id(state, action.actor_player_id)
        if discard_player is None:
            raise InvalidActionError(
                f"Unknown player ID {action.actor_player_id!r}."
            )
        players = _players_with_hands(
            state.players,
            {
                action.actor_player_id: tuple(
                    card_id
                    for card_id in discard_player.private.hand_card_ids
                    if card_id != operation.card_id
                )
            },
        )
        deck = DeckState.model_validate(
            {
                **state.deck.model_dump(),
                "hand_limit_discard_card_ids": (
                    *state.deck.hand_limit_discard_card_ids,
                    operation.card_id,
                ),
            }
        )
        return _rebuild_state(
            state,
            territories=state.territories,
            reinforcement_pools=state.reinforcement_pools,
            players=players,
            deck=deck,
            phase_id="turn_end",
            active_player_id=action.actor_player_id,
            turn_player_id=action.actor_player_id,
            decision_player_id=action.actor_player_id,
        )
    raise UnsupportedPhaseError(
        f"{operation.action_type!r} rules are not implemented yet."
    )


__all__ = [
    "InvalidChanceOutcomeError",
    "InvalidActionError",
    "UnsupportedPhaseError",
    "VictoryAdjudicationRequiredError",
    "apply_action",
    "apply_combat_outcome",
    "chance_request_for_pending_combat",
    "legal_actions",
    "resolve_pending_objective_check",
    "simulate_pending_combat_outcome",
    "validate_action",
]
