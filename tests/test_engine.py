from typing import Literal, TypedDict

import pytest

from war_game_engine.contracts import (
    Action,
    ArmyPlacement,
    ClaimTerritoryOperation,
    DeckState,
    FullGameState,
    FullPlayerState,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from war_game_engine.engine import (
    InvalidActionError,
    UnsupportedPhaseError,
    apply_action,
    legal_actions,
    validate_action,
)
from war_game_engine.randomness import RandomStreams


class ActionOverrides(TypedDict, total=False):
    actor_player_id: str
    territory_id: str
    army_count: int
    pool_id: str | None
    expected_state_version: int


def _state(
    *,
    phase_id: str = "initial_placement",
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
    territories: tuple[TerritoryState, ...] | None = None,
) -> FullGameState:
    game_territories = territories or (
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
    pools = (
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
            ),
            private=PlayerPrivateState(
                player_id=player_id,
                objective_id=f"{player_id}:objective",
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
        turn_number=1,
        active_player_id="player-1",
        turn_player_id="player-1",
        decision_player_id=decision_player_id,
        initial_placement_order=("player-1", "player-2"),
        players=players,
        territories=game_territories,
        reinforcement_pools=pools,
        deck=DeckState(
            draw_pile_card_ids=(),
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(42).snapshot(),
        is_finished=False,
    )


def _place_action(
    state: FullGameState,
    *,
    actor_player_id: str = "player-1",
    territory_id: str = "T01",
    army_count: int = 1,
    pool_id: str | None = "player-1:pool-1",
    expected_state_version: int | None = None,
    placement_phase: Literal["initial", "reinforcement"] = "initial",
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"place-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version="1.0.0",
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


def test_setup_transitions_to_complete_after_last_pool_is_spent() -> None:
    state = _state(first_pool_armies=1, second_pool_armies=1)
    state = apply_action(state, legal_actions(state)[0])
    state = apply_action(state, legal_actions(state)[0])

    assert state.phase_id == "initial_placement_complete"
    assert state.active_player_id is None
    assert state.turn_player_id is None
    assert state.decision_player_id is None
    assert state.reinforcement_pools == ()


@pytest.mark.parametrize(
    ("action_kwargs", "message"),
    (
        ({"expected_state_version": 10}, "targets state version"),
        ({"actor_player_id": "player-2"}, "not responsible for the next decision"),
        ({"territory_id": "T02"}, "does not own territory"),
        ({"territory_id": "unknown"}, "Unknown territory ID"),
        ({"army_count": 4}, "exceeds the remaining armies"),
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


def test_legal_actions_and_validation_reject_unimplemented_phases_explicitly() -> None:
    state = _state(phase_id="combat")

    with pytest.raises(UnsupportedPhaseError, match="not implemented in Phase 2"):
        legal_actions(state)


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
