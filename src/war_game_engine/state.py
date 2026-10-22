"""Public projections and player-scoped observations of full game state."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import (
    Action,
    FullGameState,
    GameEvent,
    PlayerObservation,
    PublicGameState,
)
from .versioning import OBSERVATION_SCHEMA_VERSION


def to_public_game_state(state: FullGameState) -> PublicGameState:
    return PublicGameState(
        contract_type="public_game_state",
        game_id=state.game_id,
        ruleset_id=state.ruleset_id,
        rules_version=state.rules_version,
        state_version=state.state_version,
        phase_id=state.phase_id,
        turn_number=state.turn_number,
        active_player_id=state.active_player_id,
        turn_player_id=state.turn_player_id,
        decision_player_id=state.decision_player_id,
        initial_placement_order=state.initial_placement_order,
        players=tuple(player.public for player in state.players),
        territories=state.territories,
        reinforcement_pools=state.reinforcement_pools,
        movement_control=state.movement_control,
        is_finished=state.is_finished,
    )


def observe_player(
    state: FullGameState,
    player_id: str,
    *,
    legal_actions: tuple[Action, ...] = (),
    recent_events: Iterable[GameEvent] = (),
) -> PlayerObservation:
    player = next(
        (player for player in state.players if player.public.player_id == player_id),
        None,
    )
    if player is None:
        raise ValueError(f"Unknown player ID {player_id!r}.")
    if player.public.is_eliminated:
        raise ValueError(f"Eliminated player {player_id!r} has no observation.")

    visible_events = tuple(
        event
        for event in recent_events
        if event.visibility == "public"
        or (event.visibility == "private" and player_id in event.visible_to_player_ids)
    )
    return PlayerObservation(
        contract_type="player_observation",
        game_id=state.game_id,
        player_id=player_id,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        state_version=state.state_version,
        public_state=to_public_game_state(state),
        private_state=player.private,
        legal_actions=legal_actions,
        recent_events=visible_events,
    )
