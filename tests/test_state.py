from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from war_game_engine.contracts import (
    Action,
    DeckState,
    EndTurnOperation,
    FullGameState,
    FullPlayerState,
    GameEvent,
    MovementControl,
    MovementTransfer,
    PlayerPrivateState,
    PublicGameState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.state import observe_player, to_public_game_state


def _full_state(
    *,
    state_version: int = 7,
    turn_number: int = 3,
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
        phase_id="combat",
        turn_number=turn_number,
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


def test_state_keeps_turn_player_and_decision_player_separate() -> None:
    state = _full_state(decision_player_id="player-2")

    public = to_public_game_state(state)

    assert public.turn_player_id == "player-1"
    assert public.decision_player_id == "player-2"
    assert public.active_player_id == "player-1"


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
        action_schema_version="1.0.0",
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
    with pytest.raises(ValidationError, match="Draw and discard card IDs"):
        DeckState(
            draw_pile_card_ids=("same-card",),
            discard_pile_card_ids=("same-card",),
            trade_count=0,
        )
