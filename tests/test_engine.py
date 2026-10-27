from typing import Literal, TypedDict

import pytest
from pydantic import JsonValue, ValidationError

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    AppliedChanceOutcome,
    ArmyPlacement,
    AttackOperation,
    ChanceOutcome,
    ClaimTerritoryOperation,
    DeckState,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    EndablePhaseId,
    EndPhaseOperation,
    EndTurnOperation,
    FullGameState,
    FullPlayerState,
    GamePhaseId,
    MoveArmiesOperation,
    MovementControl,
    MovementTransfer,
    ObjectiveResumePhaseId,
    OccupyTerritoryOperation,
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
from war_game_engine.engine import (
    InvalidActionError,
    InvalidChanceOutcomeError,
    UnsupportedPhaseError,
    _first_active_player,
    _next_turn_player,
    _objective_resume_phase_id,
    _players_after_elimination,
    _trade_army_bonus,
    _trade_set_is_valid,
    _transfer_eliminated_cards,
    _validate_attack,
    _validate_occupation,
    apply_action,
    apply_combat_outcome,
    chance_request_for_pending_combat,
    legal_actions,
    resolve_pending_objective_check,
    simulate_pending_combat_outcome,
    validate_action,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.versioning import ACTION_SCHEMA_VERSION


class ActionOverrides(TypedDict, total=False):
    actor_player_id: str
    territory_id: str
    pool_id: str | None
    expected_state_version: int


def _state(
    *,
    phase_id: GamePhaseId = "initial_placement",
    turn_number: int = 1,
    has_conquered_territory_this_turn: bool = False,
    decision_player_id: str = "player-1",
    first_pool_armies: int = 3,
    second_pool_armies: int = 2,
    first_pool_origin: Literal[
        "territory_count",
        "continent_bonus",
        "card_trade",
        "territory_card",
    ] = "territory_count",
    first_pool_allowed: tuple[str, ...] | None = None,
    reinforcement_pools: tuple[ReinforcementPool, ...] | None = None,
    territories: tuple[TerritoryState, ...] | None = None,
) -> FullGameState:
    if territories is not None:
        game_territories = territories
    elif phase_id == "occupation":
        game_territories = (
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
        )
    else:
        game_territories = (
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        )
    default_pools = (
        ()
        if phase_id == "occupation"
        else (
            ReinforcementPool(
                pool_id="player-1:pool-1",
                player_id="player-1",
                origin=first_pool_origin,
                remaining_armies=first_pool_armies,
                allowed_territory_ids=first_pool_allowed,
            ),
            ReinforcementPool(
                pool_id="player-2:pool-1",
                player_id="player-2",
                origin="territory_count",
                remaining_armies=second_pool_armies,
            ),
        )
    )
    pending_occupation = (
        PendingOccupation(
            contract_type="pending_occupation",
            attacker_player_id="player-1",
            defender_player_id="player-2",
            origin_territory_id="T01",
            target_territory_id="T02",
            maximum_army_count=1,
        )
        if phase_id == "occupation"
        else None
    )
    counts = {
        player_id: tuple(
            territory
            for territory in game_territories
            if territory.owner_player_id == player_id
        )
        for player_id in ("player-1", "player-2")
    }
    players = tuple(
        FullPlayerState(
            public=PublicPlayerState(
                player_id=player_id,
                color=color,
                territory_count=len(counts[player_id]),
                army_count=sum(t.army_count for t in counts[player_id]),
                is_eliminated=not counts[player_id],
                eliminated_by_player_id=(
                    "player-1"
                    if (
                        phase_id == "occupation"
                        and player_id == "player-2"
                        and not counts[player_id]
                    )
                    else None
                ),
            ),
            private=PlayerPrivateState(
                player_id=player_id,
                objective_id=(
                    "W1-OBJ-13" if player_id == "player-1" else "W1-OBJ-14"
                ),
                hand_card_ids=(),
            ),
        )
        for player_id, color in (("player-1", "blue"), ("player-2", "red"))
    )
    return FullGameState(
        contract_type="full_game_state",
        game_id="game-1",
        ruleset_id="war1-classic",
        rules_version="1.0.0",
        state_version=0,
        phase_id=phase_id,
        turn_number=turn_number,
        has_conquered_territory_this_turn=has_conquered_territory_this_turn,
        active_player_id="player-1",
        turn_player_id="player-1",
        decision_player_id=decision_player_id,
        initial_placement_order=("player-1", "player-2"),
        players=players,
        territories=game_territories,
        reinforcement_pools=(
            default_pools if reinforcement_pools is None else reinforcement_pools
        ),
        pending_occupation=pending_occupation,
        deck=DeckState(
            draw_pile_card_ids=(),
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(42).snapshot(),
        is_finished=False,
    )


def _state_with_cards(
    *,
    phase_id: GamePhaseId = "reinforcement",
    turn_number: int = 2,
    has_conquered_territory_this_turn: bool = False,
    decision_player_id: str = "player-1",
    hand_card_ids: tuple[str, ...] = (),
    opponent_hand_card_ids: tuple[str, ...] = (),
    draw_pile_card_ids: tuple[str, ...] = (),
    discard_pile_card_ids: tuple[str, ...] = (),
    hand_limit_discard_card_ids: tuple[str, ...] = (),
    trade_count: int = 0,
    territories: tuple[TerritoryState, ...] | None = None,
    reinforcement_pools: tuple[ReinforcementPool, ...] | None = None,
    with_random_streams: bool = True,
) -> FullGameState:
    if reinforcement_pools is None:
        reinforcement_pools = (
            (
                ReinforcementPool(
                    pool_id="player-1:turn-2:territory-count",
                    player_id="player-1",
                    origin="territory_count",
                    remaining_armies=3,
                ),
            )
            if phase_id == "reinforcement"
            else ()
        )
    state = _state(
        phase_id="turn_end" if phase_id == "card_discard" else phase_id,
        turn_number=turn_number,
        has_conquered_territory_this_turn=has_conquered_territory_this_turn,
        decision_player_id=decision_player_id,
        reinforcement_pools=reinforcement_pools,
        territories=territories,
    )
    players = tuple(
        FullPlayerState(
            public=player.public,
            private=PlayerPrivateState.model_validate(
                {
                    **player.private.model_dump(),
                    "hand_card_ids": (
                        hand_card_ids
                        if player.public.player_id == "player-1"
                        else opponent_hand_card_ids
                    ),
                }
            ),
        )
        for player in state.players
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": phase_id,
            "players": players,
            "deck": DeckState(
                draw_pile_card_ids=draw_pile_card_ids,
                discard_pile_card_ids=discard_pile_card_ids,
                hand_limit_discard_card_ids=hand_limit_discard_card_ids,
                trade_count=trade_count,
            ),
            "random_streams_state": (
                state.random_streams_state if with_random_streams else None
            ),
        }
    )
    return FullGameState.model_validate(state_data)


def _assert_combat_outcome_error(
    state: FullGameState,
    outcome: ChanceOutcome,
    message: str,
    *,
    random_streams_state: RandomStreamsState | None = None,
) -> None:
    with pytest.raises(InvalidChanceOutcomeError) as error_info:
        apply_combat_outcome(
            state,
            outcome,
            random_streams_state=random_streams_state,
        )

    assert str(error_info.value) == message


def _assert_invalid_action_error(
    state: FullGameState,
    message: str,
) -> None:
    with pytest.raises(InvalidActionError) as error_info:
        resolve_pending_objective_check(state)

    assert str(error_info.value) == message


def _card_action(
    state: FullGameState,
    operation: (
        TradeCardsOperation
        | DrawTerritoryCardOperation
        | DiscardTerritoryCardOperation
    ),
    *,
    actor_player_id: str = "player-1",
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"card-action-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=operation,
    )


def _attack_action(
    state: FullGameState,
    *,
    actor_player_id: str = "player-1",
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
    dice_count: int = 1,
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"attack-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=AttackOperation(
            action_type="attack",
            origin_territory_id=origin_territory_id,
            target_territory_id=target_territory_id,
            dice_count=dice_count,
        ),
    )


def _occupy_action(
    state: FullGameState,
    *,
    actor_player_id: str = "player-1",
    army_count: int = 1,
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"occupy-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=OccupyTerritoryOperation(
            action_type="occupy_territory",
            origin_territory_id=origin_territory_id,
            target_territory_id=target_territory_id,
            army_count=army_count,
        ),
    )


def _move_action(
    state: FullGameState,
    *,
    actor_player_id: str = "player-1",
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
    army_count: int = 1,
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"move-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=MoveArmiesOperation(
            action_type="move_armies",
            origin_territory_id=origin_territory_id,
            target_territory_id=target_territory_id,
            army_count=army_count,
        ),
    )


def _place_action(
    state: FullGameState,
    *,
    actor_player_id: str = "player-1",
    territory_id: str = "T01",
    army_count: Literal[1] = 1,
    pool_id: str | None = "player-1:pool-1",
    expected_state_version: int | None = None,
    placement_phase: Literal["initial", "reinforcement"] = "initial",
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"place-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=(
            state.state_version
            if expected_state_version is None
            else expected_state_version
        ),
        operation=PlaceArmiesOperation(
            action_type="place_armies",
            placements=(
                ArmyPlacement(
                    territory_id=territory_id,
                    army_count=army_count,
                    source_pool_id=pool_id,
                ),
            ),
            placement_phase=placement_phase,
        ),
    )


def _end_phase_action(
    state: FullGameState,
    phase_id: EndablePhaseId | None = None,
    *,
    actor_player_id: str = "player-1",
) -> Action:
    resolved_phase = phase_id
    if resolved_phase is None:
        if state.phase_id == "reinforcement":
            resolved_phase = "reinforcement"
        elif state.phase_id == "attack":
            resolved_phase = "attack"
        elif state.phase_id == "movement":
            resolved_phase = "movement"
        else:
            raise ValueError(f"Phase {state.phase_id!r} cannot be ended directly.")
    return Action(
        contract_type="action",
        action_id=f"end-phase-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=EndPhaseOperation(
            action_type="end_phase",
            phase_id=resolved_phase,
        ),
    )


def _end_turn_action(state: FullGameState, *, actor_player_id: str = "player-1") -> Action:
    return Action(
        contract_type="action",
        action_id=f"end-turn-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=EndTurnOperation(action_type="end_turn"),
    )


def test_legal_actions_offer_incremental_pool_specific_placements() -> None:
    state = _state(
        first_pool_armies=3,
        first_pool_origin="continent_bonus",
        first_pool_allowed=("T01",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    state_data["reinforcement_pools"] = (
        state.reinforcement_pools[0],
        ReinforcementPool(
            pool_id="player-1:territory-pool",
            player_id="player-1",
            origin="territory_count",
            remaining_armies=2,
        ),
        state.reinforcement_pools[1],
    )
    state = FullGameState.model_validate(state_data)

    actions = legal_actions(state)

    assert len(actions) == 3
    assert {
        (
            action.operation.placements[0].territory_id,
            action.operation.placements[0].source_pool_id,
        )
        for action in actions
        if isinstance(action.operation, PlaceArmiesOperation)
    } == {
        ("T01", "player-1:pool-1"),
        ("T01", "player-1:territory-pool"),
        ("T02", "player-1:territory-pool"),
    }
    assert all(action.actor_player_id == "player-1" for action in actions)
    assert all(
        action.expected_state_version == state.state_version for action in actions
    )


def test_apply_action_is_pure_and_advances_setup_players() -> None:
    initial = _state(first_pool_armies=1)
    first_action = legal_actions(initial)[0]

    after_first = apply_action(initial, first_action)

    assert initial.state_version == 0
    assert initial.territories[0].army_count == 1
    assert initial.random_streams_state == after_first.random_streams_state
    assert after_first.state_version == 1
    assert after_first.territories[0].army_count == 2
    assert after_first.active_player_id == "player-2"
    assert after_first.turn_player_id == "player-2"
    assert after_first.decision_player_id == "player-2"


@pytest.mark.rule_id("W1-INITIAL-PLACEMENT-01")
def test_initial_placement_spends_pool_across_territories_without_overdraw() -> None:
    state = _state(
        first_pool_armies=2,
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    state_data["reinforcement_pools"] = (
        state.reinforcement_pools[0],
        ReinforcementPool(
            pool_id="player-1:initial-continent",
            player_id="player-1",
            origin="continent_bonus",
            remaining_armies=1,
            allowed_territory_ids=("T02",),
        ),
        state.reinforcement_pools[1],
    )
    state = FullGameState.model_validate(state_data)

    first = apply_action(
        state,
        _place_action(state, territory_id="T01", pool_id="player-1:pool-1"),
    )
    second = apply_action(
        first,
        _place_action(first, territory_id="T02", pool_id="player-1:pool-1"),
    )
    assert second.decision_player_id == "player-1"
    assert all(
        pool.pool_id != "player-1:pool-1"
        for pool in second.reinforcement_pools
    )
    assert next(
        pool
        for pool in second.reinforcement_pools
        if pool.pool_id == "player-1:initial-continent"
    ).remaining_armies == 1

    before_overdraw = second.model_dump_json()
    with pytest.raises(InvalidActionError):
        apply_action(
            second,
            _place_action(
                second,
                territory_id="T01",
                pool_id="player-1:pool-1",
            ),
        )
    assert second.model_dump_json() == before_overdraw

    result = apply_action(
        second,
        _place_action(
            second,
            territory_id="T02",
            pool_id="player-1:initial-continent",
        ),
    )

    assert result.territories[0].army_count == 2
    assert result.territories[1].army_count == 3
    assert result.decision_player_id == "player-2"


@pytest.mark.rule_id("W1-INITIAL-PLACEMENT-01", "W1-ROUND-ORDER-01")
def test_initial_placement_starts_the_first_regular_turn() -> None:
    state = _state(first_pool_armies=1, second_pool_armies=1)
    state = apply_action(state, legal_actions(state)[0])
    state = apply_action(state, legal_actions(state)[0])

    assert state.phase_id == "reinforcement"
    assert state.turn_number == 2
    assert state.active_player_id == state.initial_placement_order[0]
    assert state.turn_player_id == state.initial_placement_order[0]
    assert state.decision_player_id == state.initial_placement_order[0]
    assert len(state.reinforcement_pools) == 1
    assert state.reinforcement_pools[0].player_id == state.turn_player_id
    assert state.reinforcement_pools[0].origin == "territory_count"
    assert state.reinforcement_pools[0].remaining_armies == 3
    assert state.reinforcement_pools[0].pool_id == (
        f"{state.turn_player_id}:turn-2:territory-count"
    )


@pytest.mark.parametrize(
    ("action_kwargs", "message"),
    (
        ({"expected_state_version": 10}, "targets state version"),
        ({"actor_player_id": "player-2"}, "not responsible for the next decision"),
        ({"territory_id": "T02"}, "does not own territory"),
        ({"territory_id": "unknown"}, "Unknown territory ID"),
        ({"pool_id": "unknown-pool"}, "not available"),
    ),
)
def test_invalid_actions_are_rejected_without_changing_state(
    action_kwargs: ActionOverrides, message: str
) -> None:
    state = _state()
    before = state.model_dump_json()
    action = _place_action(state, **action_kwargs)

    with pytest.raises(InvalidActionError, match=message):
        apply_action(state, action)

    assert state.model_dump_json() == before


def test_placement_actions_are_atomic_for_objective_checks() -> None:
    with pytest.raises(ValidationError):
        PlaceArmiesOperation.model_validate(
            {
                "action_type": "place_armies",
                "placements": ({"territory_id": "T01", "army_count": 2},),
                "placement_phase": "initial",
            }
        )
    with pytest.raises(ValidationError):
        PlaceArmiesOperation.model_validate(
            {
                "action_type": "place_armies",
                "placements": (
                    {"territory_id": "T01", "army_count": 1},
                    {"territory_id": "T02", "army_count": 1},
                ),
                "placement_phase": "initial",
            }
        )


def test_initial_placement_requires_known_pool_order() -> None:
    state_data = _state().model_dump(mode="python")
    state_data["initial_placement_order"] = ()
    state = FullGameState.model_validate(state_data)

    with pytest.raises(InvalidActionError, match="explicit player order"):
        validate_action(state, _place_action(state))


def test_unknown_board_territories_are_rejected() -> None:
    state = _state(
        territories=(
            TerritoryState(
                territory_id="T99",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        )
    )

    with pytest.raises(InvalidActionError, match="absent from BoardSpec"):
        validate_action(state, _place_action(state))


def test_legal_actions_offer_pending_occupation_choices() -> None:
    state = _state(
        phase_id="occupation",
        turn_number=2,
        reinforcement_pools=(),
    )

    actions = legal_actions(state)

    assert len(actions) == 1
    assert isinstance(actions[0].operation, OccupyTerritoryOperation)
    assert actions[0].operation.army_count == 1


def test_legal_placement_keeps_later_territories_for_restricted_pools() -> None:
    state = _state(
        first_pool_origin="continent_bonus",
        first_pool_allowed=("T02",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )

    actions = legal_actions(state)

    assert len(actions) == 1
    assert isinstance(actions[0].operation, PlaceArmiesOperation)
    assert actions[0].operation.placements[0].territory_id == "T02"


def test_reinforcement_phase_requires_prepared_pools_before_placement() -> None:
    state = _state(
        phase_id="reinforcement",
        turn_number=2,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:turn-pool",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=3,
            ),
        ),
    )

    result = apply_action(state, _end_phase_action(state))

    assert result.phase_id == "reinforcement_placement"
    assert result.decision_player_id == "player-1"
    assert result.reinforcement_pools == state.reinforcement_pools


def test_legal_actions_expose_phase_and_turn_transitions() -> None:
    reinforcement_state = _state(
        phase_id="reinforcement",
        turn_number=2,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:turn-pool",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=3,
            ),
        ),
    )
    reinforcement_action = legal_actions(reinforcement_state)[0]
    attack_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
    )
    attack_action = legal_actions(attack_state)[0]
    movement_state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
    )
    movement_action = legal_actions(movement_state)[0]
    turn_end_state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(),
    )
    turn_end_action = legal_actions(turn_end_state)[0]

    assert isinstance(reinforcement_action.operation, EndPhaseOperation)
    assert reinforcement_action.operation.phase_id == "reinforcement"
    assert isinstance(attack_action.operation, EndPhaseOperation)
    assert attack_action.operation.phase_id == "attack"
    assert isinstance(movement_action.operation, EndPhaseOperation)
    assert movement_action.operation.phase_id == "movement"
    assert isinstance(turn_end_action.operation, EndTurnOperation)


def test_reinforcement_phase_cannot_be_skipped_without_prepared_pools() -> None:
    state = _state(
        phase_id="reinforcement",
        turn_number=2,
        reinforcement_pools=(),
    )

    with pytest.raises(InvalidActionError, match="pools must be prepared"):
        apply_action(state, _end_phase_action(state))


def test_reinforcement_placement_advances_to_attacks_when_pools_are_spent() -> None:
    state = _state(
        phase_id="reinforcement_placement",
        turn_number=2,
        first_pool_armies=1,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:turn-pool",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=1,
            ),
        ),
    )

    result = apply_action(
        state,
        _place_action(
            state,
            placement_phase="reinforcement",
            pool_id="player-1:turn-pool",
        ),
    )

    assert result.phase_id == "attack"
    assert result.active_player_id == "player-1"
    assert result.decision_player_id == "player-1"
    assert result.reinforcement_pools == ()


def test_end_attack_phase_starts_final_movement() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
    )

    result = apply_action(state, _end_phase_action(state))

    assert result.phase_id == "movement"
    assert result.active_player_id == "player-1"
    assert result.turn_player_id == "player-1"
    assert result.decision_player_id == "player-1"


def test_end_phase_uses_the_decision_player_when_turn_and_active_differ() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        decision_player_id="player-2",
        reinforcement_pools=(),
    )

    result = apply_action(
        state,
        _end_phase_action(state, actor_player_id="player-2"),
    )

    assert result.phase_id == "movement"
    assert result.active_player_id == "player-2"
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


@pytest.mark.parametrize(
    ("conquered", "expected_phase"),
    ((False, "turn_end"), (True, "card_purchase")),
)
@pytest.mark.rule_id("W1-CARD-AWARD-01")
def test_end_movement_routes_by_conquest(
    conquered: bool,
    expected_phase: GamePhaseId,
) -> None:
    state = _state(
        phase_id="movement",
        turn_number=2,
        has_conquered_territory_this_turn=conquered,
        reinforcement_pools=(),
    )

    result = apply_action(state, _end_phase_action(state))

    assert result.phase_id == expected_phase
    assert result.active_player_id == "player-1"
    assert result.turn_player_id == "player-1"
    assert result.decision_player_id == "player-1"


def test_end_phase_rejects_a_stale_phase_identifier() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
    )

    with pytest.raises(InvalidActionError, match="current phase is 'attack'"):
        apply_action(state, _end_phase_action(state, "movement"))


def test_turn_end_advances_players_and_increments_round_at_wraparound() -> None:
    state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(),
    )
    state_data = state.model_dump(mode="python")
    state_data["movement_control"] = MovementControl(
        moved_transfers=(
            MovementTransfer(
                origin_territory_id="T01",
                target_territory_id="T02",
                army_count=1,
            ),
        )
    )
    state = FullGameState.model_validate(state_data)

    after_first_turn = apply_action(state, _end_turn_action(state))

    assert after_first_turn.phase_id == "reinforcement"
    assert after_first_turn.active_player_id == "player-2"
    assert after_first_turn.turn_player_id == "player-2"
    assert after_first_turn.decision_player_id == "player-2"
    assert after_first_turn.turn_number == 2
    assert after_first_turn.has_conquered_territory_this_turn is False
    assert after_first_turn.movement_control == MovementControl()
    assert after_first_turn.reinforcement_pools[0].pool_id == (
        "player-2:turn-2:territory-count"
    )

    player_two_state_data = after_first_turn.model_dump(mode="python")
    player_two_state_data.update(
        {
            "phase_id": "turn_end",
            "active_player_id": "player-2",
            "turn_player_id": "player-2",
            "decision_player_id": "player-2",
            "has_conquered_territory_this_turn": True,
            "reinforcement_pools": (),
        }
    )
    player_two_state = FullGameState.model_validate(player_two_state_data)

    after_round = apply_action(
        player_two_state,
        _end_turn_action(player_two_state, actor_player_id="player-2"),
    )

    assert after_round.active_player_id == "player-1"
    assert after_round.turn_player_id == "player-1"
    assert after_round.decision_player_id == "player-1"
    assert after_round.turn_number == 3
    assert after_round.has_conquered_territory_this_turn is False
    assert after_round.reinforcement_pools[0].pool_id == (
        "player-1:turn-3:territory-count"
    )


def test_turn_end_skips_eliminated_players() -> None:
    base_state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(),
    )
    state_data = base_state.model_dump(mode="python")
    player_one, player_two = base_state.players
    player_two_public = player_two.public.model_dump()
    player_two_public.update(
        {
            "territory_count": 0,
            "army_count": 0,
            "is_eliminated": True,
        }
    )
    player_three = FullPlayerState(
        public=PublicPlayerState(
            player_id="player-3",
            color="green",
            territory_count=1,
            army_count=1,
            is_eliminated=False,
        ),
        private=PlayerPrivateState(
            player_id="player-3",
            objective_id="player-3:objective",
            hand_card_ids=(),
        ),
    )
    state_data.update(
        {
            "players": (
                player_one,
                FullPlayerState(
                    public=PublicPlayerState.model_validate(player_two_public),
                    private=player_two.private,
                ),
                player_three,
            ),
            "territories": (
                TerritoryState(
                    territory_id="T01",
                    owner_player_id="player-1",
                    army_count=1,
                ),
                TerritoryState(
                    territory_id="T02",
                    owner_player_id="player-3",
                    army_count=1,
                ),
            ),
            "initial_placement_order": ("player-1", "player-2", "player-3"),
        }
    )
    state = FullGameState.model_validate(state_data)

    result = apply_action(state, _end_turn_action(state))

    assert result.active_player_id == "player-3"
    assert result.turn_number == 2


def test_finished_phase_is_terminal_and_has_no_legal_actions() -> None:
    state_data = _state().model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "reinforcement_pools": (),
            "winner_player_id": "player-1",
            "is_finished": True,
        }
    )
    state = FullGameState.model_validate(state_data)

    assert legal_actions(state) == ()
    with pytest.raises(InvalidActionError, match="Finished games"):
        validate_action(state, _place_action(state))


def test_finished_state_without_an_official_winner_is_rejected() -> None:
    state_data = _state().model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "reinforcement_pools": (),
            "is_finished": True,
        }
    )

    with pytest.raises(ValidationError, match="Finished games require an official"):
        FullGameState.model_validate(state_data)


@pytest.mark.parametrize(
    ("phase_id", "operation"),
    (
        (
            "reinforcement",
            TradeCardsOperation(
                action_type="trade_cards",
                card_ids=("card-1", "card-2", "card-3"),
            ),
        ),
        (
            "attack",
            AttackOperation(
                action_type="attack",
                origin_territory_id="T01",
                target_territory_id="T02",
                dice_count=1,
            ),
        ),
        (
            "occupation",
            OccupyTerritoryOperation(
                action_type="occupy_territory",
                origin_territory_id="T01",
                target_territory_id="T02",
                army_count=1,
            ),
        ),
        (
            "movement",
            MoveArmiesOperation(
                action_type="move_armies",
                origin_territory_id="T01",
                target_territory_id="T02",
                army_count=1,
            ),
        ),
    )
)
def test_rule_operations_are_guarded_by_their_phases(
    phase_id: GamePhaseId,
    operation: (
        TradeCardsOperation
        | AttackOperation
        | OccupyTerritoryOperation
        | MoveArmiesOperation
    ),
) -> None:
    wrong_phase_state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(),
    )
    wrong_phase_action = Action(
        contract_type="action",
        action_id="action-out-of-phase",
        actor_player_id="player-1",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=wrong_phase_state.state_version,
        operation=operation,
    )
    with pytest.raises(InvalidActionError, match="not legal during 'turn_end'"):
        validate_action(wrong_phase_state, wrong_phase_action)

    correct_phase_state = _state(
        phase_id=phase_id,
        turn_number=2,
        reinforcement_pools=(),
    )
    correct_phase_action = Action(
        contract_type="action",
        action_id="action-not-yet-implemented",
        actor_player_id="player-1",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=correct_phase_state.state_version,
        operation=operation,
    )

    if isinstance(operation, TradeCardsOperation):
        with pytest.raises(InvalidActionError, match="pools must be prepared"):
            validate_action(correct_phase_state, correct_phase_action)
    elif isinstance(operation, AttackOperation):
        with pytest.raises(InvalidActionError, match="at least two armies"):
            validate_action(correct_phase_state, correct_phase_action)
    elif isinstance(operation, OccupyTerritoryOperation):
        validate_action(correct_phase_state, correct_phase_action)
    elif isinstance(operation, MoveArmiesOperation):
        with pytest.raises(InvalidActionError, match="must both belong"):
            validate_action(correct_phase_state, correct_phase_action)
    else:
        with pytest.raises(UnsupportedPhaseError, match="not implemented yet"):
            validate_action(correct_phase_state, correct_phase_action)


def test_claim_actions_update_owner_and_advance_decision() -> None:
    state = _state(
        phase_id="territory_claim",
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id=None,
                army_count=0,
            ),
        ),
    )
    action = next(
        action
        for action in legal_actions(state)
        if isinstance(action.operation, ClaimTerritoryOperation)
    )

    result = apply_action(state, action)

    assert result.territories[2].owner_player_id == "player-1"
    assert result.territories[2].army_count == 1
    assert result.phase_id == "initial_placement"
    assert result.active_player_id == "player-2"
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


def test_territory_claim_uses_fallback_order_and_keeps_unclaimed_territories() -> (
    None
):
    state = _state(
        phase_id="territory_claim",
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id=None,
                army_count=0,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id=None,
                army_count=0,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    state_data["initial_placement_order"] = ()
    state = FullGameState.model_validate(state_data)
    action = next(
        action
        for action in legal_actions(state)
        if isinstance(action.operation, ClaimTerritoryOperation)
        and action.operation.territory_id == "T03"
    )

    result = apply_action(state, action)

    assert result.territories[2].owner_player_id == "player-1"
    assert result.territories[3].owner_player_id is None
    assert result.phase_id == "territory_claim"
    assert result.active_player_id == "player-2"
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


@pytest.mark.parametrize(
    ("trade_number", "expected_armies"),
    ((1, 4), (2, 6), (3, 8), (4, 10), (5, 12), (6, 15), (7, 20), (8, 25), (9, 30)),
)
@pytest.mark.rule_id("W1-TRADE-TRACK-01")
def test_card_trade_progression_is_global_and_unbounded(
    trade_number: int,
    expected_armies: int,
) -> None:
    assert _trade_army_bonus(trade_number) == expected_armies


@pytest.mark.rule_id("W1-TRADE-TRACK-01")
def test_card_trade_bonus_rejects_zero_as_a_trade_number() -> None:
    with pytest.raises(ValueError, match="one-based"):
        _trade_army_bonus(0)


@pytest.mark.parametrize(
    "card_ids",
    (
        ("T01", "T05", "T21"),
        ("T01", "T02", "T08"),
        ("JOKER-01", "T01", "T02"),
        ("T01", "JOKER-01", "T02"),
        ("T01", "T02", "JOKER-01"),
        ("JOKER-01", "JOKER-02", "T01"),
    ),
)
@pytest.mark.rule_id("W1-TRADE-SET-01")
def test_card_trade_accepts_equal_different_and_wildcard_sets(
    card_ids: tuple[str, str, str],
) -> None:
    state = _state_with_cards(hand_card_ids=card_ids)
    action = _card_action(
        state,
        TradeCardsOperation(action_type="trade_cards", card_ids=card_ids),
    )

    validate_action(state, action)


@pytest.mark.rule_id("W1-TRADE-MANDATORY-01")
def test_five_cards_leave_trade_optional() -> None:
    state = _state_with_cards(
        hand_card_ids=("T01", "T02", "T05", "T08", "T10")
    )

    actions = legal_actions(state)

    assert any(isinstance(action.operation, EndPhaseOperation) for action in actions)
    assert any(
        isinstance(action.operation, TradeCardsOperation)
        and set(action.operation.card_ids) == {"T01", "T02", "T08"}
        for action in actions
    )


@pytest.mark.parametrize(
    ("hand", "offered", "message"),
    (
        (
            ("T01", "T02", "T05"),
            ("T01", "T05", "T02"),
            "three equal or three different",
        ),
        (
            ("T01", "T02", "T05"),
            ("T01", "T01", "T02"),
            "three distinct cards",
        ),
        (
            ("T01", "T02", "T05"),
            ("T01", "T02", "T08"),
            "not in the player's hand",
        ),
    ),
)
@pytest.mark.rule_id("W1-TRADE-SET-01")
def test_invalid_card_trades_are_rejected_without_mutation(
    hand: tuple[str, ...],
    offered: tuple[str, str, str],
    message: str,
) -> None:
    state = _state_with_cards(hand_card_ids=hand)
    action = _card_action(
        state,
        TradeCardsOperation(action_type="trade_cards", card_ids=offered),
    )
    before = state.model_dump_json()

    with pytest.raises(InvalidActionError, match=message):
        apply_action(state, action)

    assert state.model_dump_json() == before


@pytest.mark.rule_id(
    "W1-TRADE-SET-01",
    "W1-TRADE-TRACK-01",
    "W1-TRADE-TERRITORY-BONUS-01",
)
def test_card_trade_creates_base_and_territory_restricted_pools() -> None:
    state = _state_with_cards(
        hand_card_ids=("T01", "T02", "JOKER-01", "T05"),
        draw_pile_card_ids=("T06",),
        trade_count=7,
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    action = _card_action(
        state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("T01", "T02", "JOKER-01"),
        ),
    )

    result = apply_action(state, action)

    assert state.deck.trade_count == 7
    assert result.deck.trade_count == 8
    assert result.deck.discard_pile_card_ids == ("T01", "T02", "JOKER-01")
    assert result.players[0].private.hand_card_ids == ("T05",)
    trade_pools = tuple(
        pool for pool in result.reinforcement_pools if pool.origin in {
            "card_trade",
            "territory_card",
        }
    )
    base_pool = next(pool for pool in trade_pools if pool.origin == "card_trade")
    territory_pools = tuple(
        pool for pool in trade_pools if pool.origin == "territory_card"
    )
    assert base_pool.remaining_armies == 25
    assert base_pool.pool_id == (
        f"player-1:turn-{state.turn_number}:trade-{state.deck.trade_count + 1}:base"
    )
    assert {
        (
            pool.source_card_id,
            pool.allowed_territory_ids,
            pool.remaining_armies,
        )
        for pool in territory_pools
    } == {
        ("T01", ("T01",), 2),
        ("T02", ("T02",), 2),
    }
    assert result.phase_id == "reinforcement"
    assert result.decision_player_id == "player-1"


@pytest.mark.rule_id("W1-TRADE-TERRITORY-BONUS-01")
def test_card_trade_requires_a_state_entry_for_each_traded_territory() -> None:
    state = _state_with_cards(
        hand_card_ids=("T01", "T05", "JOKER-01"),
    )
    action = _card_action(
        state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("T01", "T05", "JOKER-01"),
        ),
    )

    with pytest.raises(InvalidActionError, match="Game state is missing territory 'T05'"):
        apply_action(state, action)


@pytest.mark.rule_id("W1-TRADE-TERRITORY-BONUS-01")
def test_territory_card_pool_cannot_be_spent_on_another_territory() -> None:
    state = _state_with_cards(
        hand_card_ids=("T01", "T02", "JOKER-01"),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    trade_action = _card_action(
        state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("T01", "T02", "JOKER-01"),
        ),
    )
    state = apply_action(state, trade_action)
    end_reinforcement = next(
        action
        for action in legal_actions(state)
        if isinstance(action.operation, EndPhaseOperation)
    )
    state = apply_action(state, end_reinforcement)
    territory_pool = next(
        pool for pool in state.reinforcement_pools if pool.origin == "territory_card"
    )
    invalid_placement = _place_action(
        state,
        territory_id="T02" if territory_pool.allowed_territory_ids == ("T01",) else "T01",
        pool_id=territory_pool.pool_id,
        placement_phase="reinforcement",
    )

    with pytest.raises(InvalidActionError, match="not available for territory"):
        validate_action(state, invalid_placement)


@pytest.mark.rule_id("W1-REINFORCEMENT-TERR-01", "W1-REINFORCEMENT-CONT-01")
def test_end_turn_prepares_territory_and_continent_pools() -> None:
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                "player-1" if territory.continent_id == "oceania" else "player-2"
            ),
            army_count=1,
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    )
    state = _state_with_cards(
        phase_id="turn_end",
        territories=territories,
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "active_player_id": "player-2",
            "turn_player_id": "player-2",
            "decision_player_id": "player-2",
        }
    )
    state = FullGameState.model_validate(state_data)

    result = apply_action(state, _end_turn_action(state, actor_player_id="player-2"))

    assert result.phase_id == "reinforcement"
    assert result.turn_number == 3
    assert result.decision_player_id == "player-1"
    assert {
        (pool.origin, pool.remaining_armies)
        for pool in result.reinforcement_pools
    } == {("territory_count", 3), ("continent_bonus", 2)}
    continent_pool = next(
        pool
        for pool in result.reinforcement_pools
        if pool.origin == "continent_bonus"
    )
    assert continent_pool.allowed_territory_ids == ("T01", "T02", "T03", "T04")


@pytest.mark.rule_id(
    "W1-CARD-AWARD-01",
    "W1-TRADE-MANDATORY-01",
    "W1-CARD-DISCARD-RECYCLE-01",
)
def test_card_purchase_requires_conquest_and_discards_to_the_separate_pile() -> None:
    state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        hand_card_ids=("T01", "T02", "T05", "T08", "T10"),
        draw_pile_card_ids=("T12",),
        discard_pile_card_ids=("T17",),
    )
    draw_action = legal_actions(state)[0]
    assert isinstance(draw_action.operation, DrawTerritoryCardOperation)

    after_draw = apply_action(state, draw_action)

    assert after_draw.phase_id == "card_discard"
    assert len(after_draw.players[0].private.hand_card_ids) == 6
    assert after_draw.deck.draw_pile_card_ids == ()
    assert after_draw.deck.discard_pile_card_ids == ("T17",)
    assert after_draw.random_streams_state == state.random_streams_state
    discard_actions = legal_actions(after_draw)
    assert len(discard_actions) == 6
    assert all(
        isinstance(action.operation, DiscardTerritoryCardOperation)
        for action in discard_actions
    )
    discard_action = _card_action(
        after_draw,
        DiscardTerritoryCardOperation(
            action_type="discard_territory_card",
            card_id="T12",
        ),
    )

    result = apply_action(after_draw, discard_action)

    assert result.phase_id == "turn_end"
    assert len(result.players[0].private.hand_card_ids) == 5
    assert result.deck.hand_limit_discard_card_ids == ("T12",)
    assert result.deck.discard_pile_card_ids == ("T17",)


@pytest.mark.rule_id("W1-CARD-AWARD-01", "W1-TRADE-MANDATORY-01")
def test_card_purchase_without_overflow_ends_the_turn_phase() -> None:
    state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        hand_card_ids=("T01", "T02", "T05", "T08"),
        draw_pile_card_ids=("T10",),
    )

    result = apply_action(state, legal_actions(state)[0])

    assert result.phase_id == "turn_end"
    assert len(result.players[0].private.hand_card_ids) == 5


@pytest.mark.rule_id("W1-DECK-RECYCLE-01", "W1-CARD-DISCARD-RECYCLE-01")
def test_empty_draw_pile_recycles_only_traded_cards_deterministically() -> None:
    state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        discard_pile_card_ids=("T03", "JOKER-01"),
        hand_limit_discard_card_ids=("T05",),
        trade_count=1,
    )
    action = legal_actions(state)[0]

    first = apply_action(state, action)
    repeated = apply_action(state, action)

    assert first == repeated
    assert first.players[0].private.hand_card_ids[0] in ("T03", "JOKER-01")
    assert first.deck.draw_pile_card_ids in (("T03",), ("JOKER-01",))
    assert first.deck.discard_pile_card_ids == ()
    assert first.deck.hand_limit_discard_card_ids == ("T05",)
    assert first.random_streams_state != state.random_streams_state


@pytest.mark.rule_id("W1-DECK-RECYCLE-01")
def test_card_purchase_surfaces_empty_or_unrecyclable_deck() -> None:
    empty_state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
    )
    with pytest.raises(InvalidActionError, match="No territory cards are available"):
        legal_actions(empty_state)

    no_rng_state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        discard_pile_card_ids=("T03",),
        with_random_streams=False,
    )
    with pytest.raises(InvalidActionError, match="requires the saved cards random"):
        legal_actions(no_rng_state)


def test_card_purchase_rejects_unknown_card_ids() -> None:
    state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        draw_pile_card_ids=("unknown-card",),
    )

    with pytest.raises(InvalidActionError, match="Unknown territory card IDs in deck"):
        legal_actions(state)


@pytest.mark.rule_id(
    "W1-ELIMINATION-01",
    "W1-TRADE-MANDATORY-01",
    "W1-CARD-DISCARD-RECYCLE-01",
)
def test_elimination_at_five_cards_does_not_discard() -> None:
    state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T01", "T02"),
        opponent_hand_card_ids=("T03", "T04", "T05"),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    players, deck, random_streams_state = _transfer_eliminated_cards(
        state,
        "player-1",
        "player-2",
    )

    assert players[0].private.hand_card_ids == (
        "T01",
        "T02",
        "T03",
        "T04",
        "T05",
    )
    assert players[1].private.hand_card_ids == ()
    assert deck.hand_limit_discard_card_ids == ()
    assert random_streams_state == state.random_streams_state


@pytest.mark.rule_id(
    "W1-ELIMINATION-01",
    "W1-TRADE-MANDATORY-01",
    "W1-CARD-DISCARD-RECYCLE-01",
)
def test_elimination_transfers_cards_and_randomly_discards_above_five() -> None:
    state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T01", "T02", "T03", "T04"),
        opponent_hand_card_ids=("T05", "T06", "T07"),
        draw_pile_card_ids=("T08",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    first = _transfer_eliminated_cards(state, "player-1", "player-2")
    repeated = _transfer_eliminated_cards(state, "player-1", "player-2")
    players, deck, random_streams_state = first

    assert first == repeated
    assert len(players[0].private.hand_card_ids) == 5
    assert players[1].private.hand_card_ids == ()
    assert len(deck.hand_limit_discard_card_ids) == 2
    assert random_streams_state != state.random_streams_state
    assert len(set((*players[0].private.hand_card_ids, *deck.hand_limit_discard_card_ids))) == 7


@pytest.mark.rule_id("W1-ATTACK-ELIGIBILITY-01", "W1-DICE-LIMIT-01")
def test_legal_attack_actions_respect_garrisons_and_dice_limits() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    actions = legal_actions(state)
    attacks = tuple(
        action.operation
        for action in actions
        if isinstance(action.operation, AttackOperation)
    )

    assert {
        (attack.origin_territory_id, attack.target_territory_id, attack.dice_count)
        for attack in attacks
    } == {("T01", "T02", 1), ("T01", "T02", 2)}
    assert any(isinstance(action.operation, EndPhaseOperation) for action in actions)


@pytest.mark.parametrize(
    ("origin", "target", "dice_count", "message"),
    (
        ("T03", "T02", 1, "at least two armies"),
        ("T01", "T03", 1, "target must belong to an opponent"),
        ("T01", "T07", 1, "directly adjacent"),
        ("T01", "T20", 1, "directly adjacent"),
        ("T01", "T02", 2, "dice exceed the armies available"),
    ),
)
@pytest.mark.rule_id("W1-ADJACENCY-RULE-01", "W1-ATTACK-ELIGIBILITY-01")
def test_illegal_attacks_are_rejected(
    origin: str,
    target: str,
    dice_count: int,
    message: str,
) -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T07",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T20",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )

    with pytest.raises(InvalidActionError, match=message):
        validate_action(
            state,
            _attack_action(
                state,
                origin_territory_id=origin,
                target_territory_id=target,
                dice_count=dice_count,
            ),
        )


@pytest.mark.rule_id("W1-DICE-LIMIT-01", "W1-DEFENDER-DICE-CHOICE-01")
def test_attack_locks_both_dice_choices_before_chance_request() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=4,
            ),
        ),
    )
    action = _attack_action(state, dice_count=3)

    pending_state = apply_action(state, action)
    request = chance_request_for_pending_combat(pending_state)

    assert state.pending_combat is None
    assert pending_state.pending_combat is not None
    assert pending_state.pending_combat.attacker_dice_count == 3
    assert pending_state.pending_combat.defender_dice_count == 3
    assert request.request_id == pending_state.pending_combat.request_id
    assert request.request_type == "combat_dice"
    assert request.parameters == {
        "action_id": action.action_id,
        "origin_territory_id": "T01",
        "target_territory_id": "T02",
        "attacker_dice_count": 3,
        "defender_dice_count": 3,
        "dice_sides": 6,
    }
    assert legal_actions(pending_state) == ()
    with pytest.raises(InvalidActionError, match="combat outcome is required"):
        validate_action(pending_state, _end_phase_action(pending_state))


@pytest.mark.rule_id("W1-DICE-LIMIT-01", "W1-DEFENDER-DICE-CHOICE-01")
@pytest.mark.parametrize(
    ("defender_army_count", "expected_defender_dice_count"),
    ((1, 1), (2, 2), (3, 3), (4, 3), (7, 3)),
    ids=("one", "two", "three", "four", "many"),
)
def test_defender_dice_count_is_fixed_by_garrison_up_to_three(
    defender_army_count: int,
    expected_defender_dice_count: int,
) -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=defender_army_count,
            ),
        ),
    )

    pending_state = apply_action(state, _attack_action(state, dice_count=1))

    assert pending_state.pending_combat is not None
    assert (
        pending_state.pending_combat.defender_dice_count
        == expected_defender_dice_count
    )
    request = chance_request_for_pending_combat(pending_state)
    assert request.parameters["defender_dice_count"] == expected_defender_dice_count


@pytest.mark.rule_id("W1-ATTACK-SEQUENCE-01", "W1-ATTACK-CONTINUE-01")
def test_combat_outcome_applies_losses_and_allows_the_next_round() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=5,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T23",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state, dice_count=3))
    assert pending_state.pending_combat is not None
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="battle-outcome-1",
        outcome={
            "attacker_faces": [5, 4, 1],
            "defender_faces": [6, 3, 1],
        },
        visibility="public",
        outcome_source="physical_board",
    )

    result = apply_combat_outcome(pending_state, outcome)

    assert result.phase_id == "attack"
    assert result.pending_combat is None
    assert result.territories[0].army_count == 3
    assert result.territories[1].army_count == 2
    assert result.decision_player_id == "player-1"
    assert result.has_conquered_territory_this_turn is False
    assert len(result.applied_chance_outcomes) == 1
    assert result.applied_chance_outcomes[0].request_id == outcome.request_id
    assert result.applied_chance_outcomes[0].event_id == outcome.event_id
    assert result.applied_chance_outcomes[0].outcome_source == "physical_board"

    with pytest.raises(InvalidChanceOutcomeError, match="request has already"):
        apply_combat_outcome(result, outcome)

    next_pending_state = apply_action(
        result,
        _attack_action(
            result,
            origin_territory_id="T04",
            target_territory_id="T23",
        ),
    )
    assert next_pending_state.pending_combat is not None
    reused_event_id = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=next_pending_state.pending_combat.request_id,
        event_id=outcome.event_id,
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="site",
    )
    with pytest.raises(InvalidChanceOutcomeError, match="event ID has already"):
        apply_combat_outcome(next_pending_state, reused_event_id)


@pytest.mark.rule_id("W1-BATTLE-10V4-EXAMPLE-01", "W1-BATTLE-COMPARISON-01")
def test_manual_ten_against_four_example_applies_expected_losses() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=10,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=4,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state, dice_count=3))

    assert pending_state.pending_combat is not None
    assert pending_state.pending_combat.defender_dice_count == 3
    result = apply_combat_outcome(
        pending_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending_state.pending_combat.request_id,
            event_id="manual-10v4-example",
            outcome={
                "attacker_faces": [5, 4, 1],
                "defender_faces": [6, 3, 1],
            },
            visibility="public",
            outcome_source="physical_board",
        ),
    )

    assert result.territories[0].army_count == 8
    assert result.territories[1].army_count == 3


@pytest.mark.rule_id("W1-CAPTURE-01")
def test_conquest_hands_off_to_pending_occupation() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="conquest-outcome",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="site",
    )

    result = apply_combat_outcome(pending_state, outcome)

    assert result.phase_id == "occupation"
    assert result.pending_occupation is not None
    assert result.pending_occupation.origin_territory_id == "T01"
    assert result.pending_occupation.target_territory_id == "T02"
    assert result.pending_occupation.maximum_army_count == 1
    assert result.territories[1].owner_player_id is None
    assert result.territories[1].army_count == 0
    assert result.players[1].public.is_eliminated is True


@pytest.mark.rule_id("W1-CAPTURE-01")
@pytest.mark.parametrize(
    ("attacker_dice_count", "expected_maximum_army_count"),
    ((1, 1), (2, 2), (3, 3)),
)
def test_capture_transfer_limit_matches_last_attack_dice(
    attacker_dice_count: int,
    expected_maximum_army_count: int,
) -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_state = apply_action(
        state,
        _attack_action(state, dice_count=attacker_dice_count),
    )
    assert pending_state.pending_combat is not None
    conquered_state = apply_combat_outcome(
        pending_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending_state.pending_combat.request_id,
            event_id=f"capture-limit-{attacker_dice_count}",
            outcome={
                "attacker_faces": [6] * attacker_dice_count,
                "defender_faces": [1],
            },
            visibility="public",
            outcome_source="site",
        ),
    )

    assert conquered_state.pending_occupation is not None
    assert (
        conquered_state.pending_occupation.maximum_army_count
        == expected_maximum_army_count
    )
    with pytest.raises(InvalidActionError):
        apply_action(
            conquered_state,
            _occupy_action(
                conquered_state,
                army_count=expected_maximum_army_count + 1,
            ),
        )

    result = apply_action(
        conquered_state,
        _occupy_action(
            conquered_state,
            army_count=expected_maximum_army_count,
        ),
    )

    assert result.territories[0].army_count == 4 - expected_maximum_army_count
    assert result.territories[1].owner_player_id == "player-1"
    assert result.territories[1].army_count == expected_maximum_army_count


def test_simulated_combat_uses_reproducible_combat_stream() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=4,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state, dice_count=2))
    first_outcome, random_state = simulate_pending_combat_outcome(pending_state)
    repeated_outcome, repeated_random_state = simulate_pending_combat_outcome(
        pending_state
    )

    assert first_outcome == repeated_outcome
    assert random_state == repeated_random_state
    assert pending_state.pending_combat is not None
    assert first_outcome.request_id == pending_state.pending_combat.request_id
    assert first_outcome.outcome_source == "simulation"
    assert isinstance(first_outcome.outcome, dict)
    assert isinstance(first_outcome.outcome["attacker_faces"], list)
    assert isinstance(first_outcome.outcome["defender_faces"], list)
    assert len(first_outcome.outcome["attacker_faces"]) == 2
    assert len(first_outcome.outcome["defender_faces"]) == 3
    resolved = apply_combat_outcome(
        pending_state,
        first_outcome,
        random_streams_state=random_state,
    )
    assert resolved.random_streams_state == random_state


def test_observed_combat_result_cannot_advance_local_random_streams() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=2,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    observed_outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="observed-outcome",
        outcome={"attacker_faces": [6], "defender_faces": [1, 1]},
        visibility="public",
        outcome_source="physical_board",
    )

    _assert_combat_outcome_error(
        pending_state,
        observed_outcome,
        "Observed combat outcomes cannot advance internal random streams.",
        random_streams_state=state.random_streams_state,
    )


@pytest.mark.parametrize(
    ("request_id", "visibility", "outcome_data", "message"),
    (
        (
            "wrong-request",
            "public",
            {"attacker_faces": [6], "defender_faces": [1]},
            "Combat outcome request ID does not match the pending request.",
        ),
        (
            "pending",
            "private",
            {"attacker_faces": [6], "defender_faces": [1]},
            "Combat dice outcomes must be public.",
        ),
        (
            "pending",
            "public",
            {"attacker_faces": [7], "defender_faces": [1]},
            "Combat outcome must contain one to three six-sided faces for each side.",
        ),
        (
            "pending",
            "public",
            {"attacker_losses": 1, "defender_losses": 1},
            "Combat outcome must contain one to three six-sided faces for each side.",
        ),
        (
            "pending",
            "public",
            {"attacker_faces": [6, 5], "defender_faces": [1]},
            "Combat outcome face counts do not match the pending request.",
        ),
    ),
)
def test_invalid_combat_outcomes_do_not_advance_state(
    request_id: str,
    visibility: Literal["public", "private"],
    outcome_data: JsonValue,
    message: str,
) -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=2,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    resolved_request_id = (
        pending_state.pending_combat.request_id
        if request_id == "pending"
        else request_id
    )
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=resolved_request_id,
        event_id="invalid-outcome",
        outcome=outcome_data,
        visibility=visibility,
        outcome_source="site",
    )
    before = pending_state.model_dump_json()

    _assert_combat_outcome_error(pending_state, outcome, message)

    assert pending_state.model_dump_json() == before


@pytest.mark.rule_id("W1-ELIMINATION-01", "W1-TRADE-MANDATORY-01")
def test_elimination_check_transfers_cards_before_occupation() -> None:
    state = _state_with_cards(
        phase_id="attack",
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
        opponent_hand_card_ids=("T03",),
    )
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="elimination-check-outcome",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="physical_board",
    )

    before_occupation = apply_combat_outcome(pending_state, outcome)

    assert before_occupation.phase_id == "occupation"
    assert before_occupation.pending_occupation is not None
    assert before_occupation.players[1].public.eliminated_by_player_id == "player-1"
    assert before_occupation.players[1].private.hand_card_ids == ()
    assert before_occupation.players[0].private.hand_card_ids == ("T03",)
    occupy_action = legal_actions(before_occupation)[0]

    result = apply_action(before_occupation, occupy_action)

    assert result.phase_id == "attack"
    assert result.pending_objective_check is None
    assert result.territories[0].army_count == 1
    assert result.territories[1].owner_player_id == "player-1"
    assert result.territories[1].army_count == 1
    assert result.has_conquered_territory_this_turn is True


def test_elimination_overflow_discards_cards_and_advances_the_saved_rng() -> None:
    state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T03", "T04", "T05"),
        opponent_hand_card_ids=("T06", "T07", "T08"),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending = apply_action(state, _attack_action(state))
    assert pending.pending_combat is not None

    before_occupation = apply_combat_outcome(
        pending,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending.pending_combat.request_id,
            event_id="elimination-hand-overflow",
            outcome={"attacker_faces": [6], "defender_faces": [1]},
            visibility="public",
            outcome_source="physical_board",
        ),
    )

    assert before_occupation.phase_id == "occupation"
    assert len(before_occupation.players[0].private.hand_card_ids) == 5
    assert before_occupation.players[1].private.hand_card_ids == ()
    assert len(before_occupation.deck.hand_limit_discard_card_ids) == 1
    assert set(
        (
            *before_occupation.players[0].private.hand_card_ids,
            *before_occupation.deck.hand_limit_discard_card_ids,
        )
    ) == {"T03", "T04", "T05", "T06", "T07", "T08"}
    assert before_occupation.random_streams_state != state.random_streams_state
    assert before_occupation.pending_occupation is not None


@pytest.mark.rule_id("W1-ELIMINATION-01", "W1-VICTORY-01")
def test_attacker_elimination_objective_wins_before_card_transfer() -> None:
    state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T03",),
        opponent_hand_card_ids=("T04",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    player_one = state.players[0]
    state_data["players"] = (
        FullPlayerState(
            public=player_one.public,
            private=PlayerPrivateState.model_validate(
                {
                    **player_one.private.model_dump(),
                    "objective_id": "W1-OBJ-01",
                }
            ),
        ),
        state.players[1],
    )
    state = FullGameState.model_validate(state_data)
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="killer-objective-outcome",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="site",
    )

    result = apply_combat_outcome(pending_state, outcome)

    assert result.phase_id == "finished"
    assert result.winner_player_id == "player-1"
    assert result.players[0].private.hand_card_ids == ("T03",)
    assert result.players[1].private.hand_card_ids == ("T04",)
    assert result.pending_occupation is None


@pytest.mark.parametrize(
    ("target", "army_count", "message"),
    (
        ("T01", 1, "origin and target of the pending conquest"),
        ("T02", 2, "exceeds the armies available"),
    ),
)
@pytest.mark.rule_id("W1-CAPTURE-01")
def test_occupation_rejects_wrong_target_and_excess_transfer(
    target: str,
    army_count: int,
    message: str,
) -> None:
    state = _state(phase_id="occupation", turn_number=2, reinforcement_pools=())

    with pytest.raises(InvalidActionError, match=message):
        apply_action(
            state,
            _occupy_action(
                state,
                target_territory_id=target,
                army_count=army_count,
            ),
        )


@pytest.mark.rule_id(
    "W1-ATTACK-SEQUENCE-01",
    "W1-ATTACK-CONTINUE-01",
    "W1-ELIMINATION-01",
)
def test_non_eliminating_capture_resolves_objectives_before_attack_resumes() -> None:
    state = _state(
        phase_id="occupation",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    state_data["pending_occupation"] = PendingOccupation(
        contract_type="pending_occupation",
        attacker_player_id="player-1",
        defender_player_id="player-2",
        origin_territory_id="T01",
        target_territory_id="T02",
        maximum_army_count=2,
    )
    state = FullGameState.model_validate(state_data)

    result = apply_action(state, _occupy_action(state, army_count=2))

    assert result.phase_id == "attack"
    assert result.players[1].public.is_eliminated is False
    assert result.players[1].public.eliminated_by_player_id is None
    assert result.pending_objective_check is None
    assert result.territories[0].army_count == 1
    assert result.territories[1].army_count == 2
    assert result.territories[2].army_count == 1
    continued_attacks = tuple(
        action.operation
        for action in legal_actions(result)
        if isinstance(action.operation, AttackOperation)
    )
    next_attack = next(
        attack
        for attack in continued_attacks
        if attack.origin_territory_id == "T02"
        and attack.target_territory_id == "T04"
    )
    pending_state = apply_action(
        result,
        _attack_action(
            result,
            origin_territory_id=next_attack.origin_territory_id,
            target_territory_id=next_attack.target_territory_id,
            dice_count=next_attack.dice_count,
        ),
    )
    assert pending_state.pending_combat is not None
    assert pending_state.pending_combat.origin_territory_id == "T02"


@pytest.mark.rule_id("W1-VICTORY-01", "W1-CAPTURE-01")
def test_victory_is_checked_after_occupying_the_twenty_fourth_territory() -> None:
    territory_ids = tuple(
        territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
    )
    player_one_territory_ids = {"T01", *territory_ids[2:24]}
    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=(
                "player-1"
                if territory_id in player_one_territory_ids
                else "player-2"
            ),
            army_count=4 if territory_id == "T01" else 1,
        )
        for territory_id in territory_ids
    )
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=territories,
    )
    assert state.players[0].public.territory_count == 23

    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    before_occupation = apply_combat_outcome(
        pending_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending_state.pending_combat.request_id,
            event_id="twenty-fourth-territory",
            outcome={"attacker_faces": [6], "defender_faces": [1]},
            visibility="public",
            outcome_source="site",
        ),
    )
    assert before_occupation.phase_id == "occupation"
    assert before_occupation.winner_player_id is None

    result = apply_action(before_occupation, _occupy_action(before_occupation))

    assert result.phase_id == "finished"
    assert result.winner_player_id == "player-1"
    assert result.players[0].public.territory_count == 24


@pytest.mark.rule_id("W1-ELIMINATION-01", "W1-OBJECTIVE-FALLBACK-01")
def test_elimination_attributes_and_converts_other_players_objectives() -> None:
    state = _state_with_cards(
        phase_id="attack",
        opponent_hand_card_ids=("T04",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    player_three = FullPlayerState(
        public=PublicPlayerState(
            player_id="player-3",
            color="green",
            territory_count=1,
            army_count=1,
            is_eliminated=False,
        ),
        private=PlayerPrivateState(
            player_id="player-3",
            objective_id="W1-OBJ-01",
            hand_card_ids=(),
        ),
    )
    state_data.update(
        {
            "players": (*state.players, player_three),
            "territories": (
                *state.territories,
                TerritoryState(
                    territory_id="T03",
                    owner_player_id="player-3",
                    army_count=1,
                ),
            ),
            "initial_placement_order": ("player-1", "player-2", "player-3"),
        }
    )
    state = FullGameState.model_validate(state_data)
    pending_state = apply_action(state, _attack_action(state))
    assert pending_state.pending_combat is not None
    outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=pending_state.pending_combat.request_id,
        event_id="third-party-elimination",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="site",
    )

    before_occupation = apply_combat_outcome(pending_state, outcome)

    assert before_occupation.phase_id == "occupation"
    assert before_occupation.players[1].public.eliminated_by_player_id == "player-1"
    assert before_occupation.players[2].private.objective_id == "W1-OBJ-01"
    assert (
        before_occupation.players[2].private.objective_fallback_id
        == "W1-OBJ-13"
    )
    assert before_occupation.players[1].private.hand_card_ids == ()
    assert before_occupation.players[0].private.hand_card_ids == ("T04",)

    result = apply_action(
        before_occupation,
        legal_actions(before_occupation)[0],
    )
    assert result.phase_id == "attack"


@pytest.mark.rule_id("W1-END-MOVE-01", "W1-MOVEMENT-ADJACENCY-01")
def test_legal_movement_actions_are_direct_and_leave_a_garrison() -> None:
    state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )

    actions = legal_actions(state)

    moves = tuple(
        action.operation
        for action in actions
        if isinstance(action.operation, MoveArmiesOperation)
    )
    assert {
        (move.origin_territory_id, move.target_territory_id, move.army_count)
        for move in moves
    } == {
        ("T01", "T02", 1),
        ("T01", "T02", 2),
        ("T01", "T02", 3),
    }
    assert any(isinstance(action.operation, EndPhaseOperation) for action in actions)


@pytest.mark.parametrize(
    ("origin", "target", "armies", "message"),
    (
        ("T03", "T02", 1, "must both belong to the acting player"),
        ("T01", "T03", 1, "must both belong to the acting player"),
        ("T01", "T99", 1, "references an unknown territory"),
        ("T01", "T07", 1, "directly adjacent"),
        ("T01", "T01", 1, "cannot target its origin"),
        ("T01", "T02", 4, "unmoved armies or would leave no garrison"),
    ),
)
@pytest.mark.rule_id("W1-END-MOVE-01", "W1-MOVEMENT-ADJACENCY-01")
def test_invalid_movement_routes_and_counts_are_rejected(
    origin: str,
    target: str,
    armies: int,
    message: str,
) -> None:
    state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=4,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T07",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    with pytest.raises(InvalidActionError, match=message):
        apply_action(
            state,
            _move_action(
                state,
                origin_territory_id=origin,
                target_territory_id=target,
                army_count=armies,
            ),
        )


@pytest.mark.rule_id("W1-END-MOVE-01", "W1-MOVEMENT-ADJACENCY-01")
def test_movement_tracks_units_and_checks_each_direct_transfer() -> None:
    state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=5,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T23",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T05",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    first = apply_action(
        state,
        _move_action(
            state,
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=2,
        ),
    )
    assert first.phase_id == "movement"
    assert first.pending_objective_check is None
    state = first
    repeated_route = apply_action(
        state,
        _move_action(
            state,
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=1,
        ),
    )
    assert repeated_route.movement_control.moved_transfers == (
        MovementTransfer(
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=3,
        ),
    )
    state = repeated_route

    with pytest.raises(InvalidActionError, match="unmoved armies"):
        apply_action(
            state,
            _move_action(
                state,
                origin_territory_id="T02",
                target_territory_id="T04",
                army_count=2,
            ),
        )
    second = apply_action(
        state,
        _move_action(
            state,
            origin_territory_id="T02",
            target_territory_id="T04",
            army_count=1,
        ),
    )
    assert second.movement_control.moved_transfers == (
        MovementTransfer(
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=3,
        ),
        MovementTransfer(
            origin_territory_id="T02",
            target_territory_id="T04",
            army_count=1,
        ),
    )
    state = second

    with pytest.raises(InvalidActionError, match="unmoved armies"):
        apply_action(
            state,
            _move_action(
                state,
                origin_territory_id="T04",
                target_territory_id="T23",
                army_count=2,
            ),
        )
    third = apply_action(
        state,
        _move_action(
            state,
            origin_territory_id="T04",
            target_territory_id="T23",
            army_count=1,
        ),
    )

    assert third.territories[2].army_count == 1
    assert third.territories[3].army_count == 2
    assert len(third.movement_control.moved_transfers) == 3


@pytest.mark.rule_id("W1-MOVEMENT-ADJACENCY-01")
def test_movement_rejects_a_two_edge_route_without_direct_adjacency() -> None:
    edge_ids = {
        frozenset(
            (adjacency.first_territory_id, adjacency.second_territory_id)
        )
        for adjacency in WAR1_BOARD_SPEC.map.adjacencies
    }
    assert frozenset(("T01", "T03")) in edge_ids
    assert frozenset(("T03", "T20")) in edge_ids
    assert frozenset(("T01", "T20")) not in edge_ids

    state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T20",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    with pytest.raises(InvalidActionError, match="directly adjacent"):
        apply_action(
            state,
            _move_action(
                state,
                origin_territory_id="T01",
                target_territory_id="T20",
                army_count=1,
            ),
        )


@pytest.mark.rule_id("W1-VICTORY-01", "W1-OBJ-14", "W1-END-MOVE-01")
def test_final_movement_can_complete_an_objective_immediately() -> None:
    territory_ids = tuple(
        territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
    )
    player_one_territory_ids = set(territory_ids[:18])
    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=(
                "player-1"
                if territory_id in player_one_territory_ids
                else "player-2"
            ),
            army_count=(
                3
                if territory_id == "T01"
                else (
                    1
                    if territory_id == "T02"
                    else (
                        2
                        if territory_id in player_one_territory_ids
                        else 1
                    )
                )
            ),
        )
        for territory_id in territory_ids
    )
    state = _state(
        phase_id="movement",
        turn_number=2,
        reinforcement_pools=(),
        territories=territories,
    )
    player_one = state.players[0]
    player_two = state.players[1]
    state_data = state.model_dump(mode="python")
    state_data["players"] = (
        FullPlayerState(
            public=player_one.public,
            private=PlayerPrivateState.model_validate(
                {
                    **player_one.private.model_dump(),
                    "objective_id": "W1-OBJ-14",
                }
            ),
        ),
        FullPlayerState(
            public=player_two.public,
            private=PlayerPrivateState.model_validate(
                {
                    **player_two.private.model_dump(),
                    "objective_id": "W1-OBJ-12",
                }
            ),
        ),
    )
    state = FullGameState.model_validate(state_data)
    assert sum(
        territory.owner_player_id == "player-1" and territory.army_count >= 2
        for territory in state.territories
    ) == 17

    result = apply_action(
        state,
        _move_action(
            state,
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=1,
        ),
    )

    assert result.phase_id == "finished"
    assert result.winner_player_id == "player-1"
    assert result.territories[0].army_count == 2
    assert result.territories[1].army_count == 2


def test_placement_requires_an_eligible_unambiguous_pool() -> None:
    territories = (
        TerritoryState(
            territory_id="T01",
            owner_player_id="player-1",
            army_count=1,
        ),
        TerritoryState(
            territory_id="T02",
            owner_player_id="player-1",
            army_count=1,
        ),
        TerritoryState(
            territory_id="T03",
            owner_player_id="player-2",
            army_count=1,
        ),
    )
    unavailable_state = _state(
        first_pool_origin="continent_bonus",
        first_pool_allowed=("T01",),
        territories=territories,
    )
    with pytest.raises(InvalidActionError, match="No reinforcement pool"):
        validate_action(
            unavailable_state,
            _place_action(
                unavailable_state,
                territory_id="T02",
                pool_id=None,
            ),
        )

    ambiguous_data = _state(territories=territories).model_dump(mode="python")
    ambiguous_data["reinforcement_pools"] = (
        ReinforcementPool(
            pool_id="player-1:pool-1",
            player_id="player-1",
            origin="territory_count",
            remaining_armies=3,
        ),
        ReinforcementPool(
            pool_id="player-1:pool-2",
            player_id="player-1",
            origin="card_trade",
            remaining_armies=4,
        ),
        ReinforcementPool(
            pool_id="player-2:pool-1",
            player_id="player-2",
            origin="territory_count",
            remaining_armies=2,
        ),
    )
    ambiguous_state = FullGameState.model_validate(ambiguous_data)
    with pytest.raises(InvalidActionError, match="source_pool_id is required"):
        validate_action(
            ambiguous_state,
            _place_action(
                ambiguous_state,
                territory_id="T01",
                pool_id=None,
            ),
        )

    single_pool_state = _state(territories=territories)
    validate_action(
        single_pool_state,
        _place_action(
            single_pool_state,
            territory_id="T01",
            pool_id=None,
        ),
    )


def test_trade_set_rejects_a_non_three_card_tuple() -> None:
    assert not _trade_set_is_valid(("T01", "T02"), WAR1_BOARD_SPEC)


def test_elimination_player_updates_reject_unknown_or_self_eliminators() -> None:
    state = _state()
    with pytest.raises(InvalidActionError, match="invalid player"):
        _players_after_elimination(
            state.players,
            "missing-player",
            "player-1",
        )
    with pytest.raises(InvalidActionError, match="invalid player"):
        _players_after_elimination(
            state.players,
            "player-1",
            "player-1",
        )


@pytest.mark.parametrize("phase_id", ("objective_check", "finished"))
def test_objective_checks_cannot_resume_into_nonresumable_phases(
    phase_id: GamePhaseId,
) -> None:
    with pytest.raises(InvalidActionError, match="cannot resume"):
        _objective_resume_phase_id(phase_id)


def test_card_trade_rejects_unknown_cards_and_uncertified_symbols() -> None:
    unknown_card_state = _state_with_cards(
        hand_card_ids=("unknown-card", "T01", "T02")
    )
    unknown_card_action = _card_action(
        unknown_card_state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("unknown-card", "T01", "T02"),
        ),
    )
    with pytest.raises(InvalidActionError, match="Unknown territory card IDs"):
        validate_action(unknown_card_state, unknown_card_action)

    board_without_symbol = WAR1_BOARD_SPEC.map.model_copy(
        update={
            "territories": tuple(
                territory.model_copy(update={"card_symbol": None})
                if territory.territory_id == "T01"
                else territory
                for territory in WAR1_BOARD_SPEC.map.territories
            )
        }
    )
    uncertified_state = _state_with_cards(
        hand_card_ids=("T01", "T02", "JOKER-01")
    )
    uncertified_action = _card_action(
        uncertified_state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("T01", "T02", "JOKER-01"),
        ),
    )
    with pytest.raises(InvalidActionError, match="has no certified symbol"):
        validate_action(
            uncertified_state,
            uncertified_action,
            board=board_without_symbol,
        )

    custom_wildcards = ("WILD-A", "WILD-B")
    custom_deck = WAR1_BOARD_SPEC.territory_deck.model_copy(
        update={
            "card_ids": (
                *(territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories),
                *custom_wildcards,
            )
        }
    )
    custom_board = WAR1_BOARD_SPEC.model_copy(
        update={
            "wildcard_card_ids": custom_wildcards,
            "territory_deck": custom_deck,
        }
    )
    custom_wildcard_state = _state_with_cards(
        hand_card_ids=("WILD-A", "T01", "T02")
    )
    custom_wildcard_action = _card_action(
        custom_wildcard_state,
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("WILD-A", "T01", "T02"),
        ),
    )
    validate_action(
        custom_wildcard_state,
        custom_wildcard_action,
        board=custom_board,
    )


def test_draw_and_discard_operations_are_phase_and_hand_guarded() -> None:
    wrong_draw_phase = _state_with_cards(phase_id="attack", turn_number=2)
    with pytest.raises(InvalidActionError, match="not legal during 'attack'"):
        validate_action(
            wrong_draw_phase,
            _card_action(
                wrong_draw_phase,
                DrawTerritoryCardOperation(action_type="draw_territory_card"),
            ),
        )

    wrong_discard_phase = _state_with_cards(
        phase_id="attack",
        turn_number=2,
        hand_card_ids=("T01",),
    )
    with pytest.raises(InvalidActionError, match="not legal during 'attack'"):
        validate_action(
            wrong_discard_phase,
            _card_action(
                wrong_discard_phase,
                DiscardTerritoryCardOperation(
                    action_type="discard_territory_card",
                    card_id="T01",
                ),
            ),
        )

    card_purchase_state = _state_with_cards(
        phase_id="card_purchase",
        turn_number=2,
        has_conquered_territory_this_turn=True,
        hand_card_ids=("T01", "T02", "T05", "T08", "T10"),
        draw_pile_card_ids=("T12",),
    )
    card_discard_state = apply_action(
        card_purchase_state,
        legal_actions(card_purchase_state)[0],
    )
    with pytest.raises(InvalidActionError, match="not in the player's hand"):
        validate_action(
            card_discard_state,
            _card_action(
                card_discard_state,
                DiscardTerritoryCardOperation(
                    action_type="discard_territory_card",
                    card_id="JOKER-01",
                ),
            ),
        )


def test_attack_rejects_unknown_and_unowned_targets() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="unknown territory"):
        validate_action(
            state,
            _attack_action(state, target_territory_id="T99"),
        )

    unowned_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="unowned territory"):
        validate_action(unowned_state, _attack_action(unowned_state))

    wrong_origin_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=2,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="origin must belong"):
        validate_action(
            wrong_origin_state,
            _attack_action(
                wrong_origin_state,
                origin_territory_id="T03",
                target_territory_id="T02",
            ),
        )

def test_eliminated_card_transfer_rejects_invalid_players_and_missing_randomness() -> (
    None
):
    eliminated_state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T01",),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="cannot receive their own"):
        _transfer_eliminated_cards(
            eliminated_state,
            "player-1",
            "player-1",
        )
    with pytest.raises(InvalidActionError, match="not an active player"):
        _transfer_eliminated_cards(
            eliminated_state,
            "unknown-player",
            "player-2",
        )

    active_opponent_state = _state_with_cards(
        phase_id="attack",
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="has not been eliminated"):
        _transfer_eliminated_cards(
            active_opponent_state,
            "player-1",
            "player-2",
        )

    no_randomness_state = _state_with_cards(
        phase_id="attack",
        hand_card_ids=("T01", "T02", "T03"),
        opponent_hand_card_ids=("T04", "T05", "T06"),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
        with_random_streams=False,
    )
    with pytest.raises(InvalidActionError, match="requires the saved cards"):
        _transfer_eliminated_cards(
            no_randomness_state,
            "player-1",
            "player-2",
        )


def test_combat_chance_helpers_require_pending_combat_and_rng() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="no pending combat"):
        chance_request_for_pending_combat(state)
    with pytest.raises(InvalidActionError, match="no pending combat"):
        simulate_pending_combat_outcome(state)
    with pytest.raises(InvalidChanceOutcomeError, match="No combat is waiting"):
        apply_combat_outcome(
            state,
            ChanceOutcome(
                contract_type="chance_outcome",
                request_id="missing-combat",
                event_id="missing-combat-event",
                outcome={"attacker_faces": [6], "defender_faces": [1]},
                visibility="public",
                outcome_source="site",
            ),
        )

    state_data = state.model_dump(mode="python")
    state_data["random_streams_state"] = None
    state_without_randomness = FullGameState.model_validate(state_data)
    pending_state = apply_action(
        state_without_randomness,
        _attack_action(state_without_randomness),
    )
    with pytest.raises(InvalidActionError, match="requires the saved combat"):
        simulate_pending_combat_outcome(pending_state)


def test_phase_specific_validators_reject_unprepared_or_pending_actions() -> None:
    initial_state = _state()
    with pytest.raises(InvalidActionError, match="Placement phase 'reinforcement'"):
        validate_action(
            initial_state,
            _place_action(initial_state, placement_phase="reinforcement"),
        )

    reinforcement_placement_state = _state(
        phase_id="reinforcement_placement",
        turn_number=2,
    )
    with pytest.raises(
        InvalidActionError,
        match="may only use the active player's pools",
    ):
        validate_action(
            reinforcement_placement_state,
            _place_action(
                reinforcement_placement_state,
                placement_phase="reinforcement",
            ),
        )

    unprepared_trade_state = _state_with_cards(
        phase_id="reinforcement",
        hand_card_ids=("T01", "T02", "T08"),
        reinforcement_pools=(),
    )
    with pytest.raises(InvalidActionError, match="pools must be prepared"):
        validate_action(
            unprepared_trade_state,
            _card_action(
                unprepared_trade_state,
                TradeCardsOperation(
                    action_type="trade_cards",
                    card_ids=("T01", "T02", "T08"),
                ),
            ),
        )

    attack_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_combat = apply_action(attack_state, _attack_action(attack_state))
    with pytest.raises(InvalidActionError, match="combat outcome is required"):
        validate_action(pending_combat, _attack_action(pending_combat))


def test_legal_placement_validation_uses_the_custom_board() -> None:
    state = _state()
    map_without_owned_territory = WAR1_BOARD_SPEC.map.model_copy(
        update={
            "territories": tuple(
                territory
                for territory in WAR1_BOARD_SPEC.map.territories
                if territory.territory_id != "T01"
            )
        }
    )
    board_without_owned_territory = WAR1_BOARD_SPEC.model_copy(
        update={"map": map_without_owned_territory}
    )

    with pytest.raises(InvalidActionError, match="absent from BoardSpec"):
        legal_actions(state, board=board_without_owned_territory)


def test_claim_and_decision_validation_reject_unknown_or_unowned_choices() -> None:
    state = _state(
        phase_id="territory_claim",
        turn_number=1,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
        ),
    )

    def claim_action(territory_id: str) -> Action:
        return Action(
            contract_type="action",
            action_id=f"claim-{territory_id}",
            actor_player_id="player-1",
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=state.state_version,
            operation=ClaimTerritoryOperation(
                action_type="claim_territory",
                territory_id=territory_id,
            ),
        )

    with pytest.raises(InvalidActionError, match="Unknown territory ID"):
        validate_action(state, claim_action("T99"))
    with pytest.raises(InvalidActionError, match="is already owned"):
        validate_action(state, claim_action("T01"))

    no_decision_data = state.model_dump(mode="python")
    no_decision_data.update(
        {
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
        }
    )
    no_decision_state = FullGameState.model_validate(no_decision_data)
    assert legal_actions(no_decision_state) == ()
    with pytest.raises(InvalidActionError, match="no player responsible"):
        validate_action(no_decision_state, claim_action("T02"))

    map_without_unowned_territory = WAR1_BOARD_SPEC.map.model_copy(
        update={
            "territories": tuple(
                territory
                for territory in WAR1_BOARD_SPEC.map.territories
                if territory.territory_id != "T02"
            )
        }
    )
    board_without_unowned_territory = WAR1_BOARD_SPEC.model_copy(
        update={"map": map_without_unowned_territory}
    )
    with pytest.raises(InvalidActionError, match="absent from BoardSpec"):
        legal_actions(state, board=board_without_unowned_territory)


def test_legal_actions_use_the_separate_decision_player() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        decision_player_id="player-2",
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=2,
            ),
        ),
    )

    actions = legal_actions(state)
    attacks = tuple(
        action.operation
        for action in actions
        if isinstance(action.operation, AttackOperation)
    )

    assert attacks
    assert all(action.actor_player_id == "player-2" for action in actions)
    assert any(
        attack.origin_territory_id == "T02"
        and attack.target_territory_id == "T01"
        for attack in attacks
    )


def test_end_phase_and_turn_require_spent_pools_and_known_turn_order() -> None:
    attack_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:unspent",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="All reinforcement pools"):
        validate_action(attack_state, _end_phase_action(attack_state))

    turn_end_state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(),
    )
    no_order_data = turn_end_state.model_dump(mode="python")
    no_order_data["initial_placement_order"] = ()
    no_order_state = FullGameState.model_validate(no_order_data)
    with pytest.raises(InvalidActionError, match="Turn order must include"):
        validate_action(no_order_state, _end_turn_action(no_order_state))


def test_turn_end_rejects_unspent_pools_and_turn_end_outside_its_phase() -> None:
    state = _state(
        phase_id="turn_end",
        turn_number=2,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:unspent",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=1,
            ),
        ),
    )
    with pytest.raises(InvalidActionError, match="All reinforcement pools"):
        validate_action(state, _end_turn_action(state))

    attack_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
    )
    with pytest.raises(InvalidActionError, match="Turns cannot end during"):
        validate_action(attack_state, _end_turn_action(attack_state))


def test_initial_placement_search_skips_eliminated_players() -> None:
    state = _state()
    player_one, player_two = state.players
    player_one_public = player_one.public.model_dump()
    player_one_public.update(
        {"territory_count": 0, "army_count": 0, "is_eliminated": True}
    )
    player_two_public = player_two.public.model_dump()
    player_two_public.update(
        {"territory_count": 2, "army_count": 2, "is_eliminated": False}
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "initial_placement",
            "turn_number": 1,
            "active_player_id": "player-2",
            "turn_player_id": "player-2",
            "decision_player_id": "player-2",
            "players": (
                FullPlayerState(
                    public=PublicPlayerState.model_validate(player_one_public),
                    private=player_one.private,
                ),
                FullPlayerState(
                    public=PublicPlayerState.model_validate(player_two_public),
                    private=player_two.private,
                ),
            ),
            "territories": (
                TerritoryState(
                    territory_id="T01",
                    owner_player_id="player-2",
                    army_count=1,
                ),
                TerritoryState(
                    territory_id="T02",
                    owner_player_id="player-2",
                    army_count=1,
                ),
            ),
            "reinforcement_pools": (
                ReinforcementPool(
                    pool_id="player-2:initial",
                    player_id="player-2",
                    origin="territory_count",
                    remaining_armies=1,
                ),
            ),
        }
    )
    state = FullGameState.model_validate(state_data)

    assert _first_active_player(state) == "player-2"


def test_pending_objective_check_blocks_new_actions() -> None:
    state = _state(phase_id="attack", turn_number=2)
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "objective_check",
            "pending_objective_check": PendingObjectiveCheck(
                contract_type="pending_objective_check",
                actor_player_id="player-1",
                territory_id="T02",
                resume_phase_id="attack",
                resume_decision_player_id="player-1",
            ),
        }
    )
    pending_state = FullGameState.model_validate(state_data)

    with pytest.raises(InvalidActionError, match="objective check must be resolved"):
        validate_action(pending_state, _place_action(pending_state))


def _objective_check_state(
    *,
    resume_phase_id: ObjectiveResumePhaseId = "attack",
) -> FullGameState:
    state = _state(phase_id="attack", turn_number=2, reinforcement_pools=())
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "objective_check",
            "pending_objective_check": PendingObjectiveCheck(
                contract_type="pending_objective_check",
                actor_player_id="player-1",
                territory_id="T01",
                resume_phase_id=resume_phase_id,
                resume_decision_player_id="player-1",
            ),
        }
    )
    return FullGameState.model_validate(state_data)


def test_objective_resolution_requires_a_pending_check_in_its_phase() -> None:
    state = _state(phase_id="attack", turn_number=2, reinforcement_pools=())
    pending = PendingObjectiveCheck(
        contract_type="pending_objective_check",
        actor_player_id="player-1",
        territory_id="T01",
        resume_phase_id="attack",
        resume_decision_player_id="player-1",
    )

    _assert_invalid_action_error(
        state.model_copy(update={"pending_objective_check": pending}),
        "The state has no pending objective check.",
    )
    _assert_invalid_action_error(
        _objective_check_state().model_copy(
            update={"pending_objective_check": None}
        ),
        "The state has no pending objective check.",
    )


@pytest.mark.parametrize(
    ("actor_player_id", "eliminated_player_id"),
    (("player-1", "missing-player"), ("missing-player", "player-2")),
)
def test_objective_resolution_rejects_unknown_elimination_references(
    actor_player_id: str,
    eliminated_player_id: str,
) -> None:
    state = _objective_check_state()
    pending = PendingObjectiveCheck(
        contract_type="pending_objective_check",
        actor_player_id=actor_player_id,
        territory_id="T01",
        resume_phase_id="attack",
        resume_decision_player_id="player-1",
        eliminated_player_id=eliminated_player_id,
    )

    _assert_invalid_action_error(
        state.model_copy(update={"pending_objective_check": pending}),
        "Pending objective check references an unknown player.",
    )


def test_objective_resolution_clears_unrelated_pending_occupation() -> None:
    state = _state(phase_id="occupation", turn_number=2, reinforcement_pools=())
    occupation_state = state.model_copy(
        update={
            "phase_id": "objective_check",
            "pending_objective_check": PendingObjectiveCheck(
                contract_type="pending_objective_check",
                actor_player_id="player-1",
                territory_id="T01",
                resume_phase_id="attack",
                resume_decision_player_id="player-1",
            ),
        }
    )

    result = resolve_pending_objective_check(occupation_state)

    assert result.phase_id == "attack"
    assert result.pending_occupation is None
    assert result.state_version == occupation_state.state_version


def test_finished_and_unprepared_phases_do_not_offer_or_accept_actions() -> None:
    finished_data = _state().model_dump(mode="python")
    finished_data.update(
        {
            "phase_id": "finished",
            "reinforcement_pools": (),
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "winner_player_id": "player-1",
            "is_finished": True,
        }
    )
    finished_state = FullGameState.model_validate(finished_data)
    with pytest.raises(InvalidActionError, match="Finished games"):
        validate_action(finished_state, _end_phase_action(finished_state, "attack"))

    reinforcement_state = _state(
        phase_id="reinforcement",
        turn_number=2,
        reinforcement_pools=(),
    )
    with pytest.raises(UnsupportedPhaseError, match="not prepared for this turn"):
        legal_actions(reinforcement_state)

    placement_state = _state(
        phase_id="reinforcement_placement",
        turn_number=2,
        reinforcement_pools=(),
    )
    with pytest.raises(UnsupportedPhaseError, match="not prepared for this turn"):
        legal_actions(placement_state)

    unprepared_reinforcement = _state(
        phase_id="reinforcement",
        turn_number=2,
        reinforcement_pools=(),
    )
    with pytest.raises(UnsupportedPhaseError, match="not prepared for this turn"):
        legal_actions(unprepared_reinforcement)


def test_pending_action_helpers_reject_missing_or_mismatched_decisions() -> None:
    state = _state(phase_id="attack", turn_number=2, reinforcement_pools=())
    with pytest.raises(InvalidActionError, match="no pending objective check"):
        resolve_pending_objective_check(state)
    with pytest.raises(InvalidActionError, match="Turn order"):
        _next_turn_player(state, "unknown-player")

    pending_objective_data = state.model_dump(mode="python")
    pending_objective_data.update(
        {
            "phase_id": "objective_check",
            "pending_objective_check": PendingObjectiveCheck(
                contract_type="pending_objective_check",
                actor_player_id="player-1",
                territory_id="T02",
                resume_phase_id="attack",
                resume_decision_player_id="player-1",
            ),
        }
    )
    pending_objective_state = FullGameState.model_validate(pending_objective_data)
    board_without_objective = WAR1_BOARD_SPEC.model_copy(
        update={
            "objective_deck": WAR1_BOARD_SPEC.objective_deck.model_copy(
                update={
                    "card_ids": tuple(
                        objective_id
                        for objective_id in WAR1_BOARD_SPEC.objective_deck.card_ids
                        if objective_id != "W1-OBJ-13"
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="absent from BoardSpec"):
        resolve_pending_objective_check(
            pending_objective_state,
            board=board_without_objective,
        )

    attack_state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_combat = apply_action(attack_state, _attack_action(attack_state))
    pending_attack_action = _attack_action(pending_combat)
    assert isinstance(pending_attack_action.operation, AttackOperation)
    with pytest.raises(InvalidActionError, match="combat outcome is required"):
        _validate_attack(
            pending_combat,
            pending_attack_action,
            pending_attack_action.operation,
            WAR1_BOARD_SPEC,
        )

    occupation_state = _state(
        phase_id="occupation",
        turn_number=2,
        reinforcement_pools=(),
    )
    occupation_action = _occupy_action(occupation_state)
    assert isinstance(occupation_action.operation, OccupyTerritoryOperation)
    missing_pending_occupation = occupation_state.model_copy(
        update={"pending_occupation": None}
    )
    with pytest.raises(InvalidActionError, match="pending conquest"):
        _validate_occupation(
            missing_pending_occupation,
            occupation_action,
            occupation_action.operation,
        )
    wrong_attacker_action = Action(
        contract_type="action",
        action_id="wrong-occupation-actor",
        actor_player_id="player-2",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=occupation_state.state_version,
        operation=occupation_action.operation,
    )
    with pytest.raises(InvalidActionError, match="Only the attacker"):
        _validate_occupation(
            occupation_state,
            wrong_attacker_action,
            occupation_action.operation,
        )


def test_combat_outcome_rejects_blank_ids_missing_streams_and_changed_ownership() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending = apply_action(state, _attack_action(state))
    assert pending.pending_combat is not None
    request_id = pending.pending_combat.request_id

    for request_id_value, event_id_value, message in (
        (
            " ",
            "valid-event",
            "Combat outcome request ID cannot be blank.",
        ),
        (
            request_id,
            " ",
            "Combat outcome event ID cannot be blank.",
        ),
    ):
        _assert_combat_outcome_error(
            pending,
            ChanceOutcome(
                contract_type="chance_outcome",
                request_id=request_id_value,
                event_id=event_id_value,
                outcome={"attacker_faces": [6], "defender_faces": [1]},
                visibility="public",
                outcome_source="site",
            ),
            message,
        )

    simulated_outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=request_id,
        event_id="missing-updated-stream",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="simulation",
    )
    _assert_combat_outcome_error(
        pending,
        simulated_outcome,
        "Simulated combat outcomes must include the updated random stream.",
    )

    observed_outcome = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=request_id,
        event_id="observed-with-rng",
        outcome={"attacker_faces": [6], "defender_faces": [1]},
        visibility="public",
        outcome_source="site",
    )
    _assert_combat_outcome_error(
        pending,
        observed_outcome,
        "Observed combat outcomes cannot advance internal random streams.",
        random_streams_state=RandomStreams(42).snapshot(),
    )

    for changed_territory_id, changed_owner_id in (
        ("T01", "player-2"),
        ("T02", "player-1"),
    ):
        changed_territories = tuple(
            TerritoryState(
                territory_id=territory.territory_id,
                owner_player_id=(
                    changed_owner_id
                    if territory.territory_id == changed_territory_id
                    else territory.owner_player_id
                ),
                army_count=territory.army_count,
            )
            for territory in pending.territories
        )
        changed_state = pending.model_copy(
            update={"territories": changed_territories}
        )
        _assert_combat_outcome_error(
            changed_state,
            ChanceOutcome(
                contract_type="chance_outcome",
                request_id=request_id,
                event_id=f"ownership-change-{changed_territory_id}",
                outcome={"attacker_faces": [6], "defender_faces": [1]},
                visibility="public",
                outcome_source="site",
            ),
            "Territory ownership changed while combat was awaiting its outcome.",
        )


def test_combat_outcomes_reject_duplicate_ids_and_missing_pending_combat() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending = apply_action(state, _attack_action(state))
    assert pending.pending_combat is not None
    request_id = pending.pending_combat.request_id

    duplicate_request_state = pending.model_copy(
        update={
            "applied_chance_outcomes": (
                AppliedChanceOutcome(
                    request_id=request_id,
                    event_id="previous-event",
                    visibility="public",
                    outcome_source="site",
                ),
            )
        }
    )
    _assert_combat_outcome_error(
        duplicate_request_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=request_id,
            event_id="new-event",
            outcome={"attacker_faces": [6], "defender_faces": [1]},
            visibility="public",
            outcome_source="site",
        ),
        "Combat chance request has already been applied.",
    )

    duplicate_event_state = pending.model_copy(
        update={
            "applied_chance_outcomes": (
                AppliedChanceOutcome(
                    request_id="previous-request",
                    event_id="duplicate-event",
                    visibility="public",
                    outcome_source="site",
                ),
            )
        }
    )
    _assert_combat_outcome_error(
        duplicate_event_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=request_id,
            event_id="duplicate-event",
            outcome={"attacker_faces": [6], "defender_faces": [1]},
            visibility="public",
            outcome_source="site",
        ),
        "Chance outcome event ID has already been applied.",
    )

    _assert_combat_outcome_error(
        state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id="no-pending-combat",
            event_id="no-pending-event",
            outcome={"attacker_faces": [6], "defender_faces": [1]},
            visibility="public",
            outcome_source="site",
        ),
        "No combat is waiting for an outcome.",
    )


@pytest.mark.parametrize(
    ("attacker_faces", "defender_faces"),
    (
        ((6, 5), (1, 1)),
        ((6,), (1,)),
    ),
)
def test_combat_outcome_rejects_a_dice_count_mismatch_on_either_side(
    attacker_faces: tuple[int, ...],
    defender_faces: tuple[int, ...],
) -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=2,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state, dice_count=1))
    assert pending_state.pending_combat is not None

    with pytest.raises(InvalidChanceOutcomeError, match="face counts do not match"):
        apply_combat_outcome(
            pending_state,
            ChanceOutcome(
                contract_type="chance_outcome",
                request_id=pending_state.pending_combat.request_id,
                event_id="single-sided-count-mismatch",
                outcome={
                    "attacker_faces": list(attacker_faces),
                    "defender_faces": list(defender_faces),
                },
                visibility="public",
                outcome_source="site",
            ),
        )


def test_combat_loss_can_leave_exactly_one_garrison_army() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=2,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )
    pending_state = apply_action(state, _attack_action(state, dice_count=1))
    assert pending_state.pending_combat is not None

    result = apply_combat_outcome(
        pending_state,
        ChanceOutcome(
            contract_type="chance_outcome",
            request_id=pending_state.pending_combat.request_id,
            event_id="one-garrison-army-remains",
            outcome={"attacker_faces": [1], "defender_faces": [6]},
            visibility="public",
            outcome_source="site",
        ),
    )

    assert result.phase_id == "attack"
    assert result.territories[0].army_count == 1
    assert result.territories[1].owner_player_id == "player-2"


def test_attack_transition_activates_the_decision_player() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        decision_player_id="player-2",
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-1",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=3,
            ),
        ),
    )

    pending_state = apply_action(
        state,
        _attack_action(
            state,
            actor_player_id="player-2",
            origin_territory_id="T02",
            target_territory_id="T01",
        ),
    )

    assert pending_state.pending_combat is not None
    assert pending_state.active_player_id == "player-2"
    assert pending_state.turn_player_id == "player-2"
    assert pending_state.decision_player_id == "player-2"


def test_occupation_transition_activates_the_decision_player() -> None:
    state = _state(
        phase_id="attack",
        turn_number=2,
        decision_player_id="player-1",
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-2",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
            TerritoryState(
                territory_id="T04",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )
    state_data = state.model_dump(mode="python")
    state_data["phase_id"] = "occupation"
    state_data["decision_player_id"] = "player-2"
    state_data["pending_occupation"] = PendingOccupation(
        contract_type="pending_occupation",
        attacker_player_id="player-2",
        defender_player_id="player-1",
        origin_territory_id="T01",
        target_territory_id="T02",
        maximum_army_count=2,
    )
    state = FullGameState.model_validate(state_data)

    result = apply_action(
        state,
        _occupy_action(state, actor_player_id="player-2"),
    )

    assert result.phase_id == "attack"
    assert result.territories[1].owner_player_id == "player-2"
    assert result.active_player_id == "player-2"
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


def test_movement_transition_activates_the_decision_player() -> None:
    state = _state(
        phase_id="movement",
        turn_number=2,
        decision_player_id="player-2",
        reinforcement_pools=(),
        territories=(
            TerritoryState(
                territory_id="T01",
                owner_player_id="player-2",
                army_count=1,
            ),
            TerritoryState(
                territory_id="T02",
                owner_player_id="player-2",
                army_count=3,
            ),
            TerritoryState(
                territory_id="T05",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )

    result = apply_action(
        state,
        _move_action(
            state,
            actor_player_id="player-2",
            origin_territory_id="T02",
            target_territory_id="T01",
            army_count=1,
        ),
    )

    assert result.phase_id == "movement"
    assert result.active_player_id == "player-2"
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


def test_draw_and_discard_transitions_activate_the_decision_player() -> None:
    draw_state = _state_with_cards(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
        decision_player_id="player-2",
        draw_pile_card_ids=("T12",),
    )
    drawn = apply_action(
        draw_state,
        _card_action(
            draw_state,
            DrawTerritoryCardOperation(action_type="draw_territory_card"),
            actor_player_id="player-2",
        ),
    )

    assert drawn.players[1].private.hand_card_ids == ("T12",)
    assert drawn.active_player_id == "player-2"
    assert drawn.turn_player_id == "player-2"
    assert drawn.decision_player_id == "player-2"

    discard_state = _state_with_cards(
        phase_id="card_discard",
        has_conquered_territory_this_turn=True,
        decision_player_id="player-2",
        opponent_hand_card_ids=("T01", "T02", "T05", "T08", "T10", "T12"),
    )
    discarded = apply_action(
        discard_state,
        _card_action(
            discard_state,
            DiscardTerritoryCardOperation(
                action_type="discard_territory_card",
                card_id="T01",
            ),
            actor_player_id="player-2",
        ),
    )

    assert discarded.players[1].private.hand_card_ids == (
        "T02",
        "T05",
        "T08",
        "T10",
        "T12",
    )
    assert discarded.active_player_id == "player-2"
    assert discarded.turn_player_id == "player-2"
    assert discarded.decision_player_id == "player-2"
