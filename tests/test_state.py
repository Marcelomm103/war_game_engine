from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from war_game_engine.contracts import (
    Action,
    AppliedChanceOutcome,
    DeckState,
    EndTurnOperation,
    FullGameState,
    FullPlayerState,
    GameEvent,
    GamePhaseId,
    MovementControl,
    MovementTransfer,
    PendingCombat,
    PendingObjectiveCheck,
    PendingOccupation,
    PlayerPrivateState,
    PublicGameState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.state import observe_player, to_public_game_state
from war_game_engine.versioning import ACTION_SCHEMA_VERSION


def _full_state(
    *,
    state_version: int = 7,
    turn_number: int = 3,
    phase_id: GamePhaseId = "attack",
    has_conquered_territory_this_turn: bool = False,
    decision_player_id: str | None = "player-1",
) -> FullGameState:
    players = (
        FullPlayerState(
            public=PublicPlayerState(
                player_id="player-1",
                color="blue",
                territory_count=1,
                army_count=2,
                is_eliminated=False,
            ),
            private=PlayerPrivateState(
                player_id="player-1",
                objective_id="objective-1",
                hand_card_ids=("hand-card-1",),
            ),
        ),
        FullPlayerState(
            public=PublicPlayerState(
                player_id="player-2",
                color="red",
                territory_count=1,
                army_count=1,
                is_eliminated=False,
            ),
            private=PlayerPrivateState(
                player_id="player-2",
                objective_id="objective-2",
                hand_card_ids=("hand-card-2",),
            ),
        ),
    )
    territories = (
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
    )
    return FullGameState(
        contract_type="full_game_state",
        game_id="game-1",
        ruleset_id="war1-classic",
        rules_version="1.0.0",
        state_version=state_version,
        phase_id=phase_id,
        turn_number=turn_number,
        has_conquered_territory_this_turn=has_conquered_territory_this_turn,
        active_player_id="player-1",
        turn_player_id="player-1",
        decision_player_id=decision_player_id,
        initial_placement_order=("player-1", "player-2"),
        players=players,
        territories=territories,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player-1:territory-count",
                player_id="player-1",
                origin="territory_count",
                remaining_armies=3,
            ),
        ),
        deck=DeckState(
            draw_pile_card_ids=("draw-card",),
            discard_pile_card_ids=("discard-card",),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(999).snapshot(),
        is_finished=False,
    )


def _revalidate_full_state(
    state: FullGameState,
    **updates: object,
) -> FullGameState:
    data = state.model_dump(mode="python")
    data.update(updates)
    return FullGameState.model_validate(data)


def _assert_full_state_error(
    state: FullGameState,
    message: str,
    **updates: object,
) -> None:
    with pytest.raises(ValidationError) as error_info:
        _revalidate_full_state(state, **updates)

    errors = error_info.value.errors()
    assert len(errors) == 1
    context = errors[0].get("ctx")
    assert isinstance(context, dict)
    cause = context.get("error")
    assert isinstance(cause, ValueError)
    assert str(cause) == message


def _pending_combat(**updates: object) -> PendingCombat:
    data: dict[str, object] = {
        "contract_type": "pending_combat",
        "request_id": "request-1",
        "action_id": "action-1",
        "attacker_player_id": "player-1",
        "origin_territory_id": "T01",
        "target_territory_id": "T02",
        "attacker_dice_count": 1,
        "defender_dice_count": 1,
    }
    data.update(updates)
    return PendingCombat.model_validate(data)


def _pending_occupation(**updates: object) -> PendingOccupation:
    data: dict[str, object] = {
        "contract_type": "pending_occupation",
        "attacker_player_id": "player-1",
        "defender_player_id": "player-2",
        "origin_territory_id": "T01",
        "target_territory_id": "T02",
        "maximum_army_count": 1,
    }
    data.update(updates)
    return PendingOccupation.model_validate(data)


def _pending_objective_check(**updates: object) -> PendingObjectiveCheck:
    data: dict[str, object] = {
        "contract_type": "pending_objective_check",
        "actor_player_id": "player-1",
        "territory_id": "T01",
        "resume_phase_id": "attack",
        "resume_decision_player_id": "player-1",
    }
    data.update(updates)
    return PendingObjectiveCheck.model_validate(data)


def _full_state_with_attacker_target() -> FullGameState:
    state = _full_state()
    attacker = state.players[0].model_copy(
        update={
            "public": state.players[0].public.model_copy(
                update={"territory_count": 2, "army_count": 3}
            )
        }
    )
    return _revalidate_full_state(
        state,
        players=(attacker, state.players[1]),
        territories=(
            *state.territories,
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-1",
                army_count=1,
            ),
        ),
    )


def _full_state_for_occupation(
    *,
    phase_id: GamePhaseId = "occupation",
    decision_player_id: str | None = "player-1",
    pending_occupation: PendingOccupation | None = None,
    pending_objective_check: PendingObjectiveCheck | None = None,
) -> FullGameState:
    state = _full_state()
    return _revalidate_full_state(
        state,
        phase_id=phase_id,
        decision_player_id=decision_player_id,
        territories=(
            state.territories[0],
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
        pending_occupation=pending_occupation or _pending_occupation(),
        pending_objective_check=pending_objective_check,
    )


def _full_state_with_unowned_target() -> FullGameState:
    state = _full_state()
    return _revalidate_full_state(
        state,
        territories=(
            *state.territories,
            TerritoryState(
                territory_id="T03",
                owner_player_id=None,
                army_count=0,
            ),
        ),
    )


def _full_state_with_eliminated_defender() -> FullGameState:
    state = _full_state()
    defender = state.players[1].model_copy(
        update={
            "public": state.players[1].public.model_copy(
                update={
                    "territory_count": 0,
                    "army_count": 0,
                    "is_eliminated": True,
                    "eliminated_by_player_id": "player-1",
                }
            )
        }
    )
    third_player = FullPlayerState(
        public=PublicPlayerState(
            player_id="player-3",
            color="green",
            territory_count=1,
            army_count=1,
            is_eliminated=False,
        ),
        private=PlayerPrivateState(
            player_id="player-3",
            objective_id="objective-3",
            hand_card_ids=("hand-card-3",),
        ),
    )
    return _revalidate_full_state(
        state,
        players=(state.players[0], defender, third_player),
        territories=(
            state.territories[0],
            TerritoryState(
                territory_id="T02",
                owner_player_id=None,
                army_count=0,
            ),
            TerritoryState(
                territory_id="T03",
                owner_player_id="player-3",
                army_count=1,
            ),
        ),
        initial_placement_order=("player-1", "player-2", "player-3"),
    )


def _players_with_defender_eliminator(
    state: FullGameState,
    player_id: str,
) -> tuple[FullPlayerState, ...]:
    defender = state.players[1].model_copy(
        update={
            "public": state.players[1].public.model_copy(
                update={"eliminated_by_player_id": player_id}
            )
        }
    )
    return (state.players[0], defender, *state.players[2:])


def _players_with_public_updates(
    state: FullGameState,
    player_id: str,
    **updates: object,
) -> tuple[FullPlayerState, ...]:
    return tuple(
        player.model_copy(
            update={"public": player.public.model_copy(update=updates)}
        )
        if player.public.player_id == player_id
        else player
        for player in state.players
    )


def test_state_keeps_turn_player_and_decision_player_separate() -> None:
    state = _full_state(decision_player_id="player-2")

    public = to_public_game_state(state)

    assert public.turn_player_id == "player-1"
    assert public.decision_player_id == "player-2"
    assert public.active_player_id == "player-1"


def test_public_state_preserves_turn_conquest_status() -> None:
    state = _full_state(
        phase_id="card_purchase",
        has_conquered_territory_this_turn=True,
    )

    public = to_public_game_state(state)

    assert public.has_conquered_territory_this_turn is True


def test_public_state_exposes_only_public_chance_outcome_ids() -> None:
    state = _full_state()
    state_data = state.model_dump(mode="python")
    state_data["applied_chance_outcomes"] = (
        AppliedChanceOutcome(
            request_id="private-request",
            event_id="private-event",
            visibility="private",
            outcome_source="site",
        ),
        AppliedChanceOutcome(
            request_id="public-request",
            event_id="public-event",
            visibility="public",
            outcome_source="physical_board",
        ),
    )
    state_with_outcomes = FullGameState.model_validate(state_data)

    public = to_public_game_state(state_with_outcomes)

    assert tuple(outcome.event_id for outcome in public.applied_chance_outcomes) == (
        "public-event",
    )
    assert "private-event" not in public.model_dump_json()


def test_player_observation_contains_only_its_private_data() -> None:
    state = _full_state()

    observation = observe_player(state, "player-1")
    serialized = observation.model_dump_json()

    assert observation.private_state.objective_id == "objective-1"
    assert observation.private_state.hand_card_ids == ("hand-card-1",)
    assert "objective-2" not in serialized
    assert "hand-card-2" not in serialized
    assert "draw-card" not in serialized
    assert "discard-card" not in serialized
    assert "root_seed" not in serialized
    assert "bit_generator_state" not in serialized
    assert "999" not in serialized


def test_observation_filters_events_by_visibility() -> None:
    state = _full_state()
    events = (
        GameEvent(
            contract_type="game_event",
            event_id="public-event",
            game_id="game-1",
            sequence_number=1,
            state_version=7,
            rules_version="1.0.0",
            event_type="territory_claimed",
            actor_player_id="player-1",
            visibility="public",
            occurred_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
            data={},
        ),
        GameEvent(
            contract_type="game_event",
            event_id="private-to-player-1",
            game_id="game-1",
            sequence_number=2,
            state_version=7,
            rules_version="1.0.0",
            event_type="objective_drawn",
            actor_player_id="player-1",
            visibility="private",
            visible_to_player_ids=("player-1",),
            occurred_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
            data={},
        ),
        GameEvent(
            contract_type="game_event",
            event_id="private-to-player-2",
            game_id="game-1",
            sequence_number=3,
            state_version=7,
            rules_version="1.0.0",
            event_type="objective_drawn",
            actor_player_id="player-2",
            visibility="private",
            visible_to_player_ids=("player-2",),
            occurred_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
            data={},
        ),
        GameEvent(
            contract_type="game_event",
            event_id="engine-event",
            game_id="game-1",
            sequence_number=4,
            state_version=7,
            rules_version="1.0.0",
            event_type="rng_advanced",
            actor_player_id=None,
            visibility="engine",
            occurred_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
            data={},
        ),
    )

    observation = observe_player(state, "player-1", recent_events=events)

    assert tuple(event.event_id for event in observation.recent_events) == (
        "public-event",
        "private-to-player-1",
    )


def test_non_decision_player_cannot_receive_legal_actions() -> None:
    state = _full_state(decision_player_id="player-2")
    action = Action(
        contract_type="action",
        action_id="action-1",
        actor_player_id="player-1",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=EndTurnOperation(action_type="end_turn"),
    )

    with pytest.raises(ValidationError, match="responsible for the next decision"):
        observe_player(state, "player-1", legal_actions=(action,))


def test_state_counters_are_unbounded_integers() -> None:
    very_large = 2**100

    state = _full_state(state_version=very_large, turn_number=very_large)

    assert state.state_version == very_large
    assert state.turn_number == very_large


def test_public_state_rejects_inconsistent_owner_counts() -> None:
    with pytest.raises(ValidationError, match="does not match territory ownership"):
        PublicGameState(
            contract_type="public_game_state",
            game_id="game-1",
            ruleset_id="war1-classic",
            rules_version="1.0.0",
            state_version=0,
            phase_id="initial_placement",
            turn_number=1,
            active_player_id="player-1",
            players=(
                PublicPlayerState(
                    player_id="player-1",
                    color="blue",
                    territory_count=2,
                    army_count=2,
                    is_eliminated=False,
                ),
                PublicPlayerState(
                    player_id="player-2",
                    color="red",
                    territory_count=0,
                    army_count=0,
                    is_eliminated=True,
                ),
            ),
            territories=(
                TerritoryState(
                    territory_id="T01",
                    owner_player_id="player-1",
                    army_count=2,
                ),
            ),
            is_finished=False,
        )


def test_state_rejects_unknown_players_and_eliminated_owners() -> None:
    with pytest.raises(ValidationError, match="unknown player"):
        PublicGameState(
            contract_type="public_game_state",
            game_id="game-1",
            ruleset_id="war1-classic",
            rules_version="1.0.0",
            state_version=0,
            phase_id="initial_placement",
            turn_number=1,
            active_player_id=None,
            players=(
                PublicPlayerState(
                    player_id="player-1",
                    color="blue",
                    territory_count=0,
                    army_count=0,
                    is_eliminated=True,
                ),
            ),
            territories=(
                TerritoryState(
                    territory_id="T01",
                    owner_player_id="missing-player",
                    army_count=1,
                ),
            ),
            is_finished=False,
        )


def test_reinforcement_pool_and_movement_state_validate_references() -> None:
    with pytest.raises(ValidationError, match="Continent bonus pools"):
        ReinforcementPool(
            pool_id="bad-pool",
            player_id="player-1",
            origin="continent_bonus",
            remaining_armies=2,
        )
    with pytest.raises(ValidationError, match="cannot target its origin"):
        MovementControl(
            moved_transfers=(
                MovementTransfer(
                    origin_territory_id="T01",
                    target_territory_id="T01",
                    army_count=1,
                ),
            )
        )


def test_private_hands_and_deck_zones_reject_duplicate_cards() -> None:
    with pytest.raises(ValidationError, match="Hand card IDs"):
        PlayerPrivateState(
            player_id="player-1",
            objective_id="objective-1",
            hand_card_ids=("same-card", "same-card"),
        )
    with pytest.raises(ValidationError, match="Card IDs across deck zones"):
        DeckState(
            draw_pile_card_ids=("same-card",),
            discard_pile_card_ids=("same-card",),
            trade_count=0,
        )


def test_full_state_accepts_a_valid_pending_combat() -> None:
    pending_combat = _pending_combat()

    state = _revalidate_full_state(
        _full_state(),
        pending_combat=pending_combat,
    )

    assert state.pending_combat == pending_combat


@pytest.mark.parametrize(
    ("state_updates", "combat_updates", "message"),
    (
        (
            {"phase_id": "movement"},
            {},
            "Pending combat is only valid during the attack phase.",
        ),
        (
            {"decision_player_id": "player-2"},
            {},
            "Pending combat must belong to the decision player.",
        ),
        (
            {},
            {"origin_territory_id": "missing"},
            "Pending combat references an unknown territory.",
        ),
        (
            {},
            {"target_territory_id": "missing"},
            "Pending combat references an unknown territory.",
        ),
        (
            {},
            {"origin_territory_id": "T02", "target_territory_id": "T01"},
            "Pending combat origin must belong to the attacker.",
        ),
        (
            {},
            {"attacker_dice_count": 2},
            "Pending combat exceeds the attacking armies.",
        ),
        (
            {},
            {"defender_dice_count": 2},
            "Pending combat has an invalid defender dice count.",
        ),
    ),
)
def test_full_state_rejects_invalid_pending_combat(
    state_updates: dict[str, object],
    combat_updates: dict[str, object],
    message: str,
) -> None:
    state = _full_state()
    pending_combat = _pending_combat(**combat_updates)

    _assert_full_state_error(
        state,
        message,
        **state_updates,
        pending_combat=pending_combat,
    )


@pytest.mark.parametrize(
    "target_state",
    (_full_state_with_attacker_target, _full_state_with_unowned_target),
)
def test_pending_combat_target_must_belong_to_an_opponent(
    target_state: Callable[[], FullGameState],
) -> None:
    state = target_state()
    target_territory_id = "T03"

    _assert_full_state_error(
        state,
        "Pending combat target must belong to an opponent.",
        pending_combat=_pending_combat(target_territory_id=target_territory_id),
    )


def test_full_state_requires_a_pending_occupation_in_occupation_phase() -> None:
    _assert_full_state_error(
        _full_state(),
        "Occupation phase requires a pending occupation.",
        phase_id="occupation",
        pending_occupation=None,
    )


def test_full_state_accepts_a_pending_occupation_after_objective_check() -> None:
    state = _full_state_for_occupation(
        phase_id="objective_check",
        pending_objective_check=_pending_objective_check(
            territory_id="T02",
            resume_phase_id="occupation",
        ),
    )

    assert state.pending_occupation is not None
    assert state.pending_objective_check is not None


def test_pending_occupation_attributes_an_eliminated_defender_to_attacker() -> None:
    state = _full_state_with_eliminated_defender()

    occupation_state = _revalidate_full_state(
        state,
        phase_id="occupation",
        pending_occupation=_pending_occupation(),
    )

    assert occupation_state.pending_occupation is not None


def test_pending_occupation_rejects_an_eliminated_defender_with_another_eliminator() -> (
    None
):
    state = _full_state_with_eliminated_defender()
    occupation_state = _revalidate_full_state(
        state,
        phase_id="occupation",
        pending_occupation=_pending_occupation(),
    )

    _assert_full_state_error(
        occupation_state,
        "Eliminated defender must be attributed to the pending attacker.",
        players=_players_with_defender_eliminator(occupation_state, "player-3"),
    )


@pytest.mark.parametrize(
    ("state_updates", "occupation_updates", "message"),
    (
        (
            {"phase_id": "attack"},
            {},
            "Pending occupation is only valid during the occupation phase.",
        ),
        (
            {"decision_player_id": "player-2"},
            {},
            "Pending occupation must belong to the decision player.",
        ),
        (
            {},
            {"defender_player_id": "player-1"},
            "Pending occupation must reference a different defender.",
        ),
        (
            {},
            {"defender_player_id": "missing"},
            "Pending occupation must reference a different defender.",
        ),
        (
            {},
            {"origin_territory_id": "missing"},
            "Pending occupation references an unknown territory.",
        ),
        (
            {},
            {"target_territory_id": "missing"},
            "Pending occupation references an unknown territory.",
        ),
        (
            {},
            {"origin_territory_id": "T03"},
            "Pending occupation origin must belong to the attacker.",
        ),
        (
            {},
            {"target_territory_id": "T03"},
            "Pending occupation target must be unowned and empty.",
        ),
        (
            {},
            {"maximum_army_count": 2},
            "Pending occupation exceeds the available armies.",
        ),
    ),
)
def test_full_state_rejects_invalid_pending_occupation(
    state_updates: dict[str, object],
    occupation_updates: dict[str, object],
    message: str,
) -> None:
    state = _full_state_for_occupation()
    pending_occupation = _pending_occupation(**occupation_updates)

    _assert_full_state_error(
        state,
        message,
        **state_updates,
        pending_occupation=pending_occupation,
    )


def test_pending_decisions_reject_unrelated_combat_and_occupation() -> None:
    state = _full_state_for_occupation()

    _assert_full_state_error(
        state,
        "Only one game decision can be pending at a time.",
        pending_combat=_pending_combat(),
    )


def test_full_state_requires_a_pending_objective_check_in_its_phase() -> None:
    _assert_full_state_error(
        _full_state(),
        "Objective-check phase requires a pending objective check.",
        phase_id="objective_check",
        pending_objective_check=None,
    )


@pytest.mark.parametrize(
    ("state_updates", "objective_updates", "message"),
    (
        (
            {"phase_id": "attack"},
            {},
            "A pending objective check is only valid in its dedicated phase.",
        ),
        (
            {},
            {"resume_decision_player_id": "player-2"},
            "Pending objective check has an inconsistent resumption player.",
        ),
        (
            {},
            {"actor_player_id": "missing"},
            "Pending objective check references an unknown actor.",
        ),
        (
            {},
            {"territory_id": "missing"},
            "Pending objective check references an unknown territory.",
        ),
        (
            {},
            {"resume_phase_id": "occupation"},
            "Objective check before occupation requires its pending conquest.",
        ),
        (
            {},
            {"eliminated_player_id": "player-2"},
            "Pending objective check has an inconsistent elimination.",
        ),
    ),
)
def test_full_state_rejects_invalid_pending_objective_check(
    state_updates: dict[str, object],
    objective_updates: dict[str, object],
    message: str,
) -> None:
    state = _full_state()
    state_updates = {"phase_id": "objective_check", **state_updates}

    _assert_full_state_error(
        state,
        message,
        **state_updates,
        pending_objective_check=_pending_objective_check(**objective_updates),
    )


def test_pending_occupation_must_match_the_objective_check_conquest() -> None:
    state = _full_state_for_occupation(
        phase_id="objective_check",
        pending_objective_check=_pending_objective_check(
            territory_id="T02",
            resume_phase_id="occupation",
        ),
    )
    _assert_full_state_error(
        state,
        "Objective check before occupation requires its pending conquest.",
        pending_objective_check=_pending_objective_check(
            territory_id="T03",
            resume_phase_id="occupation",
        ),
    )


def test_pending_objective_check_rejects_an_unknown_eliminated_player() -> None:
    _assert_full_state_error(
        _full_state(),
        "Pending objective check has an inconsistent elimination.",
        phase_id="objective_check",
        pending_objective_check=_pending_objective_check(
            eliminated_player_id="missing-player",
        ),
    )


def test_pending_objective_check_rejects_an_eliminator_other_than_its_actor() -> None:
    state = _full_state_with_eliminated_defender()
    objective_state = _revalidate_full_state(
        state,
        phase_id="objective_check",
        pending_objective_check=_pending_objective_check(
            territory_id="T02",
            eliminated_player_id="player-2",
        ),
    )

    _assert_full_state_error(
        objective_state,
        "Pending objective check has an inconsistent elimination.",
        players=_players_with_defender_eliminator(objective_state, "player-3"),
    )


def test_full_state_requires_unique_player_colors_and_territory_ids() -> None:
    state = _full_state()

    _assert_full_state_error(
        state,
        "Player IDs must be unique; duplicates: player-1.",
        players=(state.players[0], state.players[0]),
    )
    _assert_full_state_error(
        state,
        "Player colors must be unique; duplicates: blue.",
        players=_players_with_public_updates(state, "player-2", color=" BLUE "),
    )
    _assert_full_state_error(
        state,
        "Territory state IDs must be unique; duplicates: T01.",
        territories=(state.territories[0], state.territories[0]),
    )


def test_full_state_enforces_territory_ownership_and_player_counts() -> None:
    state = _full_state()

    _assert_full_state_error(
        state,
        "Territory 'T01' references unknown player 'missing'.",
        territories=(
            state.territories[0].model_copy(
                update={"owner_player_id": "missing"}
            ),
            state.territories[1],
        ),
    )
    _assert_full_state_error(
        state,
        "Unowned territories cannot contain armies.",
        territories=(
            state.territories[0].model_copy(
                update={"owner_player_id": None}
            ),
            state.territories[1],
        ),
    )
    _assert_full_state_error(
        state,
        "Eliminated players cannot own territories.",
        players=_players_with_public_updates(
            state,
            "player-1",
            is_eliminated=True,
        ),
    )
    _assert_full_state_error(
        state,
        "Owned territories must contain at least one army.",
        territories=(
            state.territories[0].model_copy(update={"army_count": 0}),
            state.territories[1],
        ),
    )
    _assert_full_state_error(
        state,
        "Territory count for player 'player-1' does not match territory ownership.",
        players=_players_with_public_updates(
            state,
            "player-1",
            territory_count=2,
        ),
    )
    _assert_full_state_error(
        state,
        "Army count for player 'player-1' does not match territory army counts.",
        players=_players_with_public_updates(
            state,
            "player-1",
            army_count=3,
        ),
    )
    _assert_full_state_error(
        state,
        "Elimination status for player 'player-2' does not match territory ownership.",
        players=_players_with_public_updates(
            state,
            "player-2",
            territory_count=0,
            army_count=0,
        ),
        territories=(
            state.territories[0],
            state.territories[1].model_copy(
                update={"owner_player_id": None, "army_count": 0}
            ),
        ),
    )


def test_full_state_validates_eliminator_and_decision_player_references() -> None:
    state = _full_state()

    _assert_full_state_error(
        state,
        "Only eliminated players can have an eliminator.",
        players=_players_with_public_updates(
            state,
            "player-1",
            eliminated_by_player_id="player-2",
        ),
    )
    for eliminator_id, message in (
        ("player-2", "A player cannot eliminate themselves."),
        ("missing-player", "Eliminator must reference a known player."),
    ):
        _assert_full_state_error(
            state,
            message,
            players=_players_with_public_updates(
                state,
                "player-2",
                territory_count=0,
                army_count=0,
                is_eliminated=True,
                eliminated_by_player_id=eliminator_id,
            ),
            territories=(
                state.territories[0],
                state.territories[1].model_copy(
                    update={"owner_player_id": None, "army_count": 0}
                ),
            ),
        )
    _assert_full_state_error(
        state,
        "active_player_id must match turn_player_id when both exist.",
        turn_player_id="player-2",
    )
    _assert_full_state_error(
        state,
        "The turn player must reference a known, active player.",
        active_player_id=None,
        turn_player_id="missing-player",
        decision_player_id=None,
    )
    _assert_full_state_error(
        state,
        "The decision player must reference a known, active player.",
        decision_player_id="missing-player",
    )
    _assert_full_state_error(
        _full_state_with_eliminated_defender(),
        "The decision player must reference a known, active player.",
        decision_player_id="player-2",
    )


def test_full_state_validates_initial_placement_order() -> None:
    state = _full_state()

    _assert_full_state_error(
        state,
        "Initial placement order must be unique; duplicates: player-1.",
        initial_placement_order=("player-1", "player-1"),
    )
    _assert_full_state_error(
        state,
        "Initial placement order references unknown players: missing-player.",
        initial_placement_order=("player-1", "missing-player"),
    )
    _assert_full_state_error(
        state,
        "Initial placement order must include every player.",
        initial_placement_order=("player-1",),
    )


def test_full_state_validates_reinforcement_pools_and_movement_ledger() -> None:
    state = _full_state()
    pool = state.reinforcement_pools[0]

    _assert_full_state_error(
        state,
        "Reinforcement pool IDs must be unique; duplicates: player-1:territory-count.",
        reinforcement_pools=(pool, pool),
    )
    _assert_full_state_error(
        state,
        "Reinforcement pool 'player-1:territory-count' references an inactive player.",
        reinforcement_pools=(
            pool.model_copy(update={"player_id": "missing-player"}),
        ),
    )
    eliminated_state = _full_state_with_eliminated_defender()
    _assert_full_state_error(
        eliminated_state,
        "Reinforcement pool 'player-1:territory-count' references an inactive player.",
        reinforcement_pools=(
            pool.model_copy(update={"player_id": "player-2"}),
        ),
    )
    _assert_full_state_error(
        state,
        "Reinforcement pool 'restricted' references unknown territories: missing.",
        reinforcement_pools=(
            pool.model_copy(
                update={
                    "pool_id": "restricted",
                    "allowed_territory_ids": ("missing",),
                }
            ),
        ),
    )
    _assert_full_state_error(
        state,
        "Movement control references an unknown territory.",
        movement_control=MovementControl(
            moved_transfers=(
                MovementTransfer(
                    origin_territory_id="missing",
                    target_territory_id="T02",
                    army_count=1,
                ),
            )
        ),
    )
    _assert_full_state_error(
        state,
        "Movement ledger exceeds armies in territory 'T02'.",
        movement_control=MovementControl(
            moved_transfers=(
                MovementTransfer(
                    origin_territory_id="T01",
                    target_territory_id="T02",
                    army_count=2,
                ),
            )
        ),
    )


@pytest.mark.parametrize(
    "phase_id",
    (
        "reinforcement",
        "reinforcement_placement",
        "attack",
        "occupation",
        "movement",
        "card_purchase",
        "card_discard",
        "turn_end",
    ),
)
def test_regular_turn_phases_require_round_two(phase_id: GamePhaseId) -> None:
    _assert_full_state_error(
        _full_state(),
        "Regular-turn phases cannot occur before round two.",
        phase_id=phase_id,
        turn_number=1,
    )


@pytest.mark.parametrize("phase_id", ("card_purchase", "card_discard"))
def test_card_phases_require_a_conquest_this_turn(phase_id: GamePhaseId) -> None:
    _assert_full_state_error(
        _full_state(),
        "Card purchase and discard phases require a territory conquest this turn.",
        phase_id=phase_id,
        turn_number=3,
        has_conquered_territory_this_turn=False,
    )


def test_finished_phase_and_flag_must_agree() -> None:
    state = _full_state()

    _assert_full_state_error(
        state,
        "The finished phase and is_finished flag must agree.",
        phase_id="finished",
    )
    _assert_full_state_error(
        state,
        "The finished phase and is_finished flag must agree.",
        is_finished=True,
    )


def test_finished_games_have_no_active_decision_and_retain_a_winner() -> None:
    _assert_full_state_error(
        _full_state(),
        "Finished games cannot have an active player decision.",
        phase_id="finished",
        is_finished=True,
    )

    finished = _revalidate_full_state(
        _full_state(),
        phase_id="finished",
        is_finished=True,
        active_player_id=None,
        turn_player_id=None,
        decision_player_id=None,
        winner_player_id="player-1",
    )

    assert finished.is_finished
    assert finished.winner_player_id == "player-1"
