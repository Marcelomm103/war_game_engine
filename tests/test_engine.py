from typing import Literal, TypedDict

import pytest
from pydantic import JsonValue, ValidationError

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
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
    MovementTransfer,
    OccupyTerritoryOperation,
    PendingOccupation,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
    TradeCardsOperation,
)
from war_game_engine.engine import (
    InvalidActionError,
    InvalidChanceOutcomeError,
    UnsupportedPhaseError,
    _trade_army_bonus,
    _transfer_eliminated_cards,
    apply_action,
    apply_combat_outcome,
    chance_request_for_pending_combat,
    legal_actions,
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
        phase_id=phase_id,
        turn_number=turn_number,
        has_conquered_territory_this_turn=has_conquered_territory_this_turn,
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
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
    dice_count: int = 1,
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"attack-{state.state_version}",
        actor_player_id="player-1",
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
    army_count: int = 1,
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"occupy-{state.state_version}",
        actor_player_id="player-1",
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
    origin_territory_id: str = "T01",
    target_territory_id: str = "T02",
    army_count: int = 1,
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"move-{state.state_version}",
        actor_player_id="player-1",
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
        actor_player_id="player-1",
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
    assert after_first.decision_player_id == "player-2"


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


@pytest.mark.parametrize(
    ("conquered", "expected_phase"),
    ((False, "turn_end"), (True, "card_purchase")),
)
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

    after_first_turn = apply_action(state, _end_turn_action(state))

    assert after_first_turn.phase_id == "reinforcement"
    assert after_first_turn.active_player_id == "player-2"
    assert after_first_turn.turn_number == 2
    assert after_first_turn.has_conquered_territory_this_turn is False

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
    assert after_round.turn_number == 3
    assert after_round.has_conquered_territory_this_turn is False


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
    assert result.turn_player_id == "player-2"
    assert result.decision_player_id == "player-2"


@pytest.mark.parametrize(
    ("trade_number", "expected_armies"),
    ((1, 4), (2, 6), (3, 8), (4, 10), (5, 12), (6, 15), (7, 20), (8, 25), (9, 30)),
)
def test_card_trade_progression_is_global_and_unbounded(
    trade_number: int,
    expected_armies: int,
) -> None:
    assert _trade_army_bonus(trade_number) == expected_armies


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
def test_card_trade_accepts_equal_different_and_wildcard_sets(
    card_ids: tuple[str, str, str],
) -> None:
    state = _state_with_cards(hand_card_ids=card_ids)
    action = _card_action(
        state,
        TradeCardsOperation(action_type="trade_cards", card_ids=card_ids),
    )

    validate_action(state, action)


def test_five_cards_leave_trade_optional() -> None:
    state = _state_with_cards(
        hand_card_ids=("T01", "T02", "T05", "T08", "T10")
    )

    actions = legal_actions(state)

    assert any(isinstance(action.operation, EndPhaseOperation) for action in actions)


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
    assert {(pool.origin, pool.remaining_armies) for pool in trade_pools} == {
        ("card_trade", 25),
        ("territory_card", 2),
    }
    territory_pool = next(pool for pool in trade_pools if pool.origin == "territory_card")
    assert territory_pool.source_card_id == "T01"
    assert territory_pool.allowed_territory_ids == ("T01",)
    assert result.phase_id == "reinforcement"
    assert result.decision_player_id == "player-1"


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
        ("T01", "T02", 2, "dice exceed the armies available"),
    ),
)
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

    next_pending_state = apply_action(result, _attack_action(result))
    assert next_pending_state.pending_combat is not None
    reused_event_id = ChanceOutcome(
        contract_type="chance_outcome",
        request_id=next_pending_state.pending_combat.request_id,
        event_id=outcome.event_id,
        outcome={"attacker_faces": [6], "defender_faces": [1, 1]},
        visibility="public",
        outcome_source="site",
    )
    with pytest.raises(InvalidChanceOutcomeError, match="event ID has already"):
        apply_combat_outcome(next_pending_state, reused_event_id)


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

    with pytest.raises(
        InvalidChanceOutcomeError,
        match="Observed combat outcomes cannot advance",
    ):
        apply_combat_outcome(
            pending_state,
            observed_outcome,
            random_streams_state=state.random_streams_state,
        )


@pytest.mark.parametrize(
    ("request_id", "visibility", "outcome_data", "message"),
    (
        (
            "wrong-request",
            "public",
            {"attacker_faces": [6], "defender_faces": [1]},
            "request ID does not match",
        ),
        (
            "pending",
            "private",
            {"attacker_faces": [6], "defender_faces": [1]},
            "must be public",
        ),
        (
            "pending",
            "public",
            {"attacker_faces": [7], "defender_faces": [1]},
            "one to three six-sided faces",
        ),
        (
            "pending",
            "public",
            {"attacker_losses": 1, "defender_losses": 1},
            "one to three six-sided faces",
        ),
        (
            "pending",
            "public",
            {"attacker_faces": [6, 5], "defender_faces": [1]},
            "face counts do not match",
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

    with pytest.raises(InvalidChanceOutcomeError, match=message):
        apply_combat_outcome(pending_state, outcome)

    assert pending_state.model_dump_json() == before


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


def test_non_eliminating_capture_resolves_objectives_before_attack_resumes() -> None:
    state = _state(
        phase_id="occupation",
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
                owner_player_id=None,
                army_count=0,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-2",
                army_count=1,
            ),
        ),
    )

    result = apply_action(state, _occupy_action(state))

    assert result.phase_id == "attack"
    assert result.players[1].public.is_eliminated is False
    assert result.players[1].public.eliminated_by_player_id is None
    assert result.pending_objective_check is None


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
    } == {("T01", "T02", 1)}
    assert any(isinstance(action.operation, EndPhaseOperation) for action in actions)


@pytest.mark.parametrize(
    ("origin", "target", "armies", "message"),
    (
        ("T03", "T02", 1, "must both belong to the acting player"),
        ("T01", "T03", 1, "must both belong to the acting player"),
        ("T01", "T07", 1, "directly adjacent"),
        ("T01", "T01", 1, "cannot target its origin"),
        ("T01", "T02", 4, "unmoved armies or would leave no garrison"),
    ),
)
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
