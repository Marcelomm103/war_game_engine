import json
from datetime import datetime, timezone
from typing import Literal, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from war_game_engine.contracts import (
    Action,
    ChanceOutcome,
    ContractEnvelope,
    FullGameState,
    FullReplay,
    IncompatibleContractVersionError,
    IndividualEliminationOutcome,
    OfficialVictoryOutcome,
    OperationalInterruptionOutcome,
    PlayerObservation,
    PublicGameState,
    PublicReplay,
    ReplayActionEntry,
    ReplayChanceEntry,
    ReplayOutcome,
    ResignationOutcome,
    Snapshot,
    TrainingTruncationOutcome,
    parse_contract_json,
)
from war_game_engine.engine import apply_action, legal_actions
from war_game_engine.randomness import RandomStreams
from war_game_engine.replay import (
    IncompatibleReplaySchemaError,
    IncompatibleSnapshotSchemaError,
    ReplayDivergenceError,
    SnapshotRestoreError,
    create_full_replay,
    create_snapshot,
    replay_full_replay,
    replay_public_replay,
    restore_snapshot,
    sanitize_public_replay,
)
from war_game_engine.serialization import canonical_json_bytes, canonical_sha256
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.state import observe_player, to_public_game_state
from war_game_engine.versioning import API_VERSION, CONTRACT_SCHEMA_VERSION


def _prepared_state() -> FullGameState:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )
    return prepare_game("replay-game", players, 17).state


def _public_replay_inputs() -> tuple[
    FullGameState,
    FullReplay,
    Snapshot,
    Snapshot,
    PublicGameState,
]:
    state = _prepared_state()
    full_snapshot = create_snapshot(
        "public-sanitization-full",
        state,
        visibility="full",
    )
    public_state = to_public_game_state(state)
    public_snapshot = create_snapshot(
        "public-sanitization-public",
        public_state,
        visibility="public",
    )
    full_replay = create_full_replay(
        "public-sanitization-source",
        full_snapshot,
        (),
        state,
    )
    return state, full_replay, full_snapshot, public_snapshot, public_state


def _assert_public_sanitization_error(
    full_replay: FullReplay,
    public_initial_snapshot: Snapshot,
    public_final_state: PublicGameState,
    message: str,
) -> None:
    with pytest.raises(ValueError) as error_info:
        sanitize_public_replay(
            full_replay,
            "public-sanitization-result",
            public_initial_snapshot,
            public_final_state,
        )

    assert str(error_info.value) == message


def _assert_snapshot_error(
    snapshot_id: str,
    state: FullGameState | PublicGameState | PlayerObservation,
    visibility: Literal["public", "full", "player_private"],
    message: str,
    *,
    subject_player_id: str | None = None,
) -> None:
    with pytest.raises(SnapshotRestoreError) as error_info:
        create_snapshot(
            snapshot_id,
            state,
            visibility=visibility,
            subject_player_id=subject_player_id,
        )

    assert str(error_info.value) == message


def _state_with_individual_elimination() -> tuple[FullGameState, str]:
    state = _prepared_state()
    eliminated_player_id = state.players[-1].public.player_id
    recipient_player_id = state.players[0].public.player_id
    territories = tuple(
        territory.model_copy(update={"owner_player_id": recipient_player_id})
        if territory.owner_player_id == eliminated_player_id
        else territory
        for territory in state.territories
    )
    territory_counts = {
        player.public.player_id: 0 for player in state.players
    }
    army_counts = {player.public.player_id: 0 for player in state.players}
    for territory in territories:
        if territory.owner_player_id is None:
            continue
        territory_counts[territory.owner_player_id] += 1
        army_counts[territory.owner_player_id] += territory.army_count
    players = tuple(
        player.model_copy(
            update={
                "public": player.public.model_copy(
                    update={
                        "territory_count": territory_counts[
                            player.public.player_id
                        ],
                        "army_count": army_counts[player.public.player_id],
                        "is_eliminated": (
                            territory_counts[player.public.player_id] == 0
                        ),
                        "eliminated_by_player_id": (
                            recipient_player_id
                            if player.public.player_id == eliminated_player_id
                            else None
                        ),
                    }
                )
            }
        )
        for player in state.players
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "attack",
            "turn_number": 2,
            "active_player_id": recipient_player_id,
            "turn_player_id": recipient_player_id,
            "decision_player_id": recipient_player_id,
            "reinforcement_pools": (),
            "players": players,
            "territories": territories,
        }
    )
    return FullGameState.model_validate(state_data), eliminated_player_id


def _update_random_state(state: FullGameState, streams: RandomStreams) -> FullGameState:
    data = state.model_dump(mode="python")
    data["engine_random_state"] = None
    data["random_streams_state"] = streams.snapshot()
    return FullGameState.model_validate(data)


def test_canonical_json_hash_is_stable_across_model_round_trip() -> None:
    state = _prepared_state()
    state_json = state.model_dump_json()
    restored = FullGameState.model_validate_json(state_json)

    assert canonical_sha256(state) == canonical_sha256(restored)
    assert canonical_sha256(state).startswith("sha256:")
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'


def test_canonical_json_serializes_model_timestamps() -> None:
    snapshot = create_snapshot(
        "canonical-time",
        _prepared_state(),
        visibility="full",
        captured_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
    )

    serialized = canonical_json_bytes(snapshot)

    assert b'"captured_at":"2026-10-23T00:00:00Z"' in serialized


def test_full_snapshot_restores_state_and_all_random_streams() -> None:
    state = _prepared_state()
    snapshot = create_snapshot("full-1", state, visibility="full")

    restored = restore_snapshot(snapshot)

    assert restored.state == state
    assert restored.random_streams is not None
    assert state.random_streams_state is not None
    original_streams = RandomStreams.restore(state.random_streams_state)
    assert restored.random_streams.die("combat") == original_streams.die("combat")
    assert restored.random_streams.policy_generator("policy-1").integers(0, 1000) == (
        original_streams.policy_generator("policy-1").integers(0, 1000)
    )


def test_public_and_player_private_snapshots_do_not_contain_rng_or_other_players() -> (
    None
):
    state = _prepared_state()
    public_snapshot = create_snapshot(
        "public-1",
        to_public_game_state(state),
        visibility="public",
    )
    player_id = state.decision_player_id
    assert player_id is not None
    private_snapshot = create_snapshot(
        "private-1",
        observe_player(state, player_id),
        visibility="player_private",
        subject_player_id=player_id,
    )

    public_json = public_snapshot.model_dump_json()
    private_json = private_snapshot.model_dump_json()

    assert "root_seed" not in public_json
    assert "bit_generator_state" not in public_json
    assert "root_seed" not in private_json
    assert "bit_generator_state" not in private_json
    assert isinstance(restore_snapshot(public_snapshot).state, PublicGameState)
    assert isinstance(restore_snapshot(private_snapshot).state, PlayerObservation)
    assert restore_snapshot(public_snapshot).random_streams is None
    assert restore_snapshot(private_snapshot).random_streams is None


def test_full_snapshot_requires_typed_random_state() -> None:
    state = _prepared_state()
    data = state.model_dump(mode="python")
    data["engine_random_state"] = {"legacy": "not-restorable"}
    data["random_streams_state"] = None
    legacy_state = FullGameState.model_validate(data)

    _assert_snapshot_error(
        "legacy-full",
        legacy_state,
        "full",
        "Full snapshots require a restorable random stream state.",
    )


def test_snapshot_creation_rejects_incompatible_visibility_and_state_types() -> None:
    state = _prepared_state()
    public_state = to_public_game_state(state)
    player_id = state.decision_player_id
    assert player_id is not None
    player_observation = observe_player(state, player_id)

    _assert_snapshot_error(
        "wrong-full-state",
        public_state,
        "full",
        "Full snapshots require FullGameState.",
    )
    _assert_snapshot_error(
        "wrong-public-state",
        state,
        "public",
        "Public snapshots require PublicGameState.",
    )
    _assert_snapshot_error(
        "public-subject",
        public_state,
        "public",
        "Public snapshots cannot have a subject player.",
        subject_player_id=player_id,
    )
    _assert_snapshot_error(
        "wrong-private-state",
        public_state,
        "player_private",
        "Player-private snapshots require PlayerObservation.",
        subject_player_id=player_id,
    )
    _assert_snapshot_error(
        "private-subject-mismatch",
        player_observation,
        "player_private",
        "Private snapshot subject must match its player.",
        subject_player_id="another-player",
    )
    invalid_visibility = cast(
        Literal["public", "full", "player_private"],
        "internal",
    )
    with pytest.raises(SnapshotRestoreError) as error_info:
        create_snapshot(
            "unsupported-visibility",
            public_state,
            visibility=invalid_visibility,
        )
    assert str(error_info.value) == "Unsupported snapshot visibility 'internal'."


def test_snapshot_hash_and_replay_schema_are_checked() -> None:
    snapshot = create_snapshot("full-2", _prepared_state(), visibility="full")
    invalid_hash_data = snapshot.model_dump(mode="python")
    invalid_hash_data["state_hash"] = f"sha256:{'0' * 64}"
    with pytest.raises(ValidationError, match="hash does not match"):
        Snapshot.model_validate(invalid_hash_data)

    incompatible_data = snapshot.model_dump(mode="python")
    incompatible_data["replay_version"] = "2.0.0"
    incompatible = Snapshot.model_validate(incompatible_data)
    with pytest.raises(IncompatibleSnapshotSchemaError, match="Unsupported snapshot"):
        restore_snapshot(incompatible)

    tampered = create_snapshot("tampered", _prepared_state(), visibility="full")
    tampered_state = tampered.state
    assert isinstance(tampered_state, FullGameState)
    assert tampered_state.random_streams_state is not None
    stream_state = tampered_state.random_streams_state.streams[0]
    stream_state.bit_generator_state["tampered"] = True
    with pytest.raises(SnapshotRestoreError, match="no longer matches"):
        restore_snapshot(tampered)


def test_full_replay_reexecutes_actions_and_chance_outcomes() -> None:
    initial_state = _prepared_state()
    initial_snapshot = create_snapshot(
        "replay-initial",
        initial_state,
        visibility="full",
        captured_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )
    action = legal_actions(initial_state)[0]
    after_action = apply_action(initial_state, action)
    assert initial_state.random_streams_state is not None
    final_streams = RandomStreams.restore(initial_state.random_streams_state)
    private_die = final_streams.die("combat")
    public_die = final_streams.die("combat")
    final_state = _update_random_state(after_action, final_streams)
    entries = (
        ReplayActionEntry(
            entry_type="action",
            sequence_number=1,
            action=action,
        ),
        ReplayChanceEntry(
            entry_type="chance_outcome",
            sequence_number=2,
            outcome=ChanceOutcome(
                contract_type="chance_outcome",
                request_id="private-draw",
                event_id="private-outcome",
                outcome=private_die,
                visibility="private",
                outcome_source="simulation",
                visible_to_player_ids=("player-1",),
            ),
        ),
        ReplayChanceEntry(
            entry_type="chance_outcome",
            sequence_number=3,
            outcome=ChanceOutcome(
                contract_type="chance_outcome",
                request_id="public-dice",
                event_id="public-outcome",
                outcome=public_die,
                visibility="public",
                outcome_source="simulation",
            ),
        ),
    )
    replay = create_full_replay(
        "full-replay",
        initial_snapshot,
        entries,
        final_state,
    )

    def apply_action_entry(
        state: FullGameState, replay_action: Action, _streams: RandomStreams
    ) -> FullGameState:
        return apply_action(state, replay_action)

    def apply_chance_entry(
        state: FullGameState,
        outcome: ChanceOutcome,
        streams: RandomStreams,
    ) -> FullGameState:
        assert streams.die("combat") == outcome.outcome
        return state

    result = replay_full_replay(
        replay,
        apply_action=apply_action_entry,
        apply_chance_outcome=apply_chance_entry,
    )

    assert result.state == final_state
    assert result.random_streams is not None
    assert result.random_streams.snapshot() == final_state.random_streams_state
    assert result.outcome is None

    envelope = ContractEnvelope(
        schema_version=CONTRACT_SCHEMA_VERSION,
        api_version=API_VERSION,
        payload=replay,
    )
    assert parse_contract_json(envelope.model_dump_json()).payload == replay

    incompatible_action = json.loads(envelope.model_dump_json())
    incompatible_action["payload"]["entries"][0]["action"]["action_schema_version"] = (
        "2.0.0"
    )
    with pytest.raises(IncompatibleContractVersionError, match="action schema"):
        parse_contract_json(json.dumps(incompatible_action))


def test_public_replay_is_separate_and_drops_private_chance_outcomes() -> None:
    initial_state = _prepared_state()
    public_action = legal_actions(initial_state)[0]
    initial_full_snapshot = create_snapshot(
        "full-public-source",
        initial_state,
        visibility="full",
    )
    public_initial_snapshot = create_snapshot(
        "public-replay-initial",
        to_public_game_state(initial_state),
        visibility="public",
    )
    entries = (
        ReplayActionEntry(
            entry_type="action",
            sequence_number=1,
            action=public_action,
        ),
        ReplayChanceEntry(
            entry_type="chance_outcome",
            sequence_number=2,
            outcome=ChanceOutcome(
                contract_type="chance_outcome",
                request_id="secret-card",
                event_id="private-card",
                outcome="objective-01",
                visibility="private",
                outcome_source="simulation",
                visible_to_player_ids=("player-1",),
            ),
        ),
        ReplayChanceEntry(
            entry_type="chance_outcome",
            sequence_number=3,
            outcome=ChanceOutcome(
                contract_type="chance_outcome",
                request_id="visible-roll",
                event_id="public-roll",
                outcome=[4, 5],
                visibility="public",
                outcome_source="simulation",
                visible_to_player_ids=("player-1",),
            ),
        ),
    )
    full_replay = create_full_replay(
        "full-source",
        initial_full_snapshot,
        entries,
        initial_state,
        outcome=TrainingTruncationOutcome(
            outcome_type="training_truncation",
            limit_type="time_budget_ms",
            limit_value=30_000,
        ),
    )
    public_state = to_public_game_state(initial_state)
    public_replay = sanitize_public_replay(
        full_replay,
        "public-view",
        public_initial_snapshot,
        public_state,
    )

    assert isinstance(public_replay, PublicReplay)
    assert public_replay.outcome == full_replay.outcome
    assert len(public_replay.entries) == 2
    public_action_entry = public_replay.entries[0]
    assert isinstance(public_action_entry, ReplayActionEntry)
    assert public_action_entry.sequence_number == 1
    assert public_action_entry.action == public_action
    public_entry = public_replay.entries[1]
    assert isinstance(public_entry, ReplayChanceEntry)
    assert public_entry.sequence_number == 2
    assert public_entry.outcome.event_id == "public-roll"
    assert public_entry.outcome.outcome_source == "simulation"
    serialized = public_replay.model_dump_json()
    assert "private-card" not in serialized
    assert "objective-01" not in serialized
    assert "root_seed" not in serialized

    seen_actions: list[Action] = []

    def public_action_is_observable(
        state: PublicGameState,
        action: Action,
    ) -> PublicGameState:
        assert action == public_action
        seen_actions.append(action)
        return state

    def public_chance_is_observable(
        state: PublicGameState, outcome: ChanceOutcome
    ) -> PublicGameState:
        assert outcome.visibility == "public"
        return state

    result = replay_public_replay(
        public_replay,
        apply_action=public_action_is_observable,
        apply_chance_outcome=public_chance_is_observable,
    )
    assert result.state == public_state
    assert result.random_streams is None
    assert result.outcome == full_replay.outcome
    assert seen_actions == [public_action]

    def identity_action(state: PublicGameState, _action: Action) -> PublicGameState:
        return state

    def identity_chance(
        state: PublicGameState,
        _outcome: ChanceOutcome,
    ) -> PublicGameState:
        return state

    for changed_field, changed_value in (
        ("game_id", "other-game"),
        ("ruleset_id", "other-ruleset"),
        ("rules_version", "2.0.0"),
    ):

        def mismatched_reducer(
            state: PublicGameState,
            _action: Action,
        ) -> PublicGameState:
            return state.model_copy(update={changed_field: changed_value})

        with pytest.raises(ReplayDivergenceError) as error_info:
            replay_public_replay(
                public_replay,
                apply_action=mismatched_reducer,
                apply_chance_outcome=identity_chance,
            )
        assert str(error_info.value) == (
            "Replay reducer returned a state for another game or ruleset."
        )

    invalid_hash_replay = public_replay.model_copy(
        update={"final_state_hash": f"sha256:{'0' * 64}"}
    )
    with pytest.raises(ReplayDivergenceError) as hash_error_info:
        replay_public_replay(
            invalid_hash_replay,
            apply_action=identity_action,
            apply_chance_outcome=identity_chance,
        )
    assert str(hash_error_info.value) == (
        "Public replay did not reproduce the recorded final state hash."
    )

    invalid_outcome_replay = public_replay.model_copy(
        update={
            "outcome": OfficialVictoryOutcome(
                outcome_type="official_victory",
                winner_player_id="player-1",
            )
        }
    )
    with pytest.raises(ReplayDivergenceError) as outcome_error_info:
        replay_public_replay(
            invalid_outcome_replay,
            apply_action=identity_action,
            apply_chance_outcome=identity_chance,
        )
    assert str(outcome_error_info.value) == (
        "Public replay outcome does not match its final game state."
    )

    invalid_snapshot_replay = public_replay.model_copy(
        update={"initial_snapshot": initial_full_snapshot}
    )
    with pytest.raises(SnapshotRestoreError) as restore_error_info:
        replay_public_replay(
            invalid_snapshot_replay,
            apply_action=identity_action,
            apply_chance_outcome=identity_chance,
        )
    assert str(restore_error_info.value) == (
        "Public replay could not restore public state."
    )

    envelope = ContractEnvelope(
        schema_version=CONTRACT_SCHEMA_VERSION,
        api_version=API_VERSION,
        payload=public_replay,
    )
    assert parse_contract_json(envelope.model_dump_json()).payload == public_replay


def test_public_replay_sanitization_validates_visibility_and_state_types() -> None:
    state, full_replay, full_snapshot, public_snapshot, public_state = (
        _public_replay_inputs()
    )

    _assert_public_sanitization_error(
        full_replay.model_copy(update={"initial_snapshot": public_snapshot}),
        public_snapshot,
        public_state,
        "Only a full replay can be sanitized.",
    )

    invalid_full_snapshot = full_snapshot.model_copy(update={"state": public_state})
    _assert_public_sanitization_error(
        full_replay.model_copy(update={"initial_snapshot": invalid_full_snapshot}),
        public_snapshot,
        public_state,
        "Full replay initial snapshot must contain FullGameState.",
    )

    _assert_public_sanitization_error(
        full_replay,
        full_snapshot,
        public_state,
        "Sanitized replays require a public initial snapshot.",
    )

    invalid_public_snapshot = public_snapshot.model_copy(update={"state": state})
    _assert_public_sanitization_error(
        full_replay,
        invalid_public_snapshot,
        public_state,
        "Public replay snapshot must contain PublicGameState.",
    )


def test_public_replay_sanitization_validates_snapshot_and_final_state_identity() -> (
    None
):
    _, full_replay, _, public_snapshot, public_state = _public_replay_inputs()

    other_game_state = public_state.model_copy(update={"game_id": "other-game"})
    other_game_snapshot = create_snapshot(
        "public-sanitization-other-game",
        other_game_state,
        visibility="public",
    )
    _assert_public_sanitization_error(
        full_replay,
        other_game_snapshot,
        public_state,
        "Public snapshots must match the full replay game ID.",
    )

    inconsistent_initial_state = public_state.model_copy(
        update={"state_version": public_state.state_version + 1}
    )
    inconsistent_initial_snapshot = create_snapshot(
        "public-sanitization-inconsistent-initial",
        inconsistent_initial_state,
        visibility="public",
    )
    _assert_public_sanitization_error(
        full_replay,
        inconsistent_initial_snapshot,
        public_state,
        "Public initial snapshot does not match the full snapshot.",
    )

    for changed_field, changed_value in (
        ("ruleset_id", "other-ruleset"),
        ("rules_version", "2.0.0"),
    ):
        inconsistent_final_state = public_state.model_copy(
            update={changed_field: changed_value}
        )
        _assert_public_sanitization_error(
            full_replay,
            public_snapshot,
            inconsistent_final_state,
            "Public final state does not match the full replay.",
        )


def test_public_replay_sanitization_validates_the_replay_outcome() -> None:
    _, full_replay, _, public_snapshot, public_state = _public_replay_inputs()
    invalid_replay = full_replay.model_copy(
        update={
            "outcome": OfficialVictoryOutcome(
                outcome_type="official_victory",
                winner_player_id="player-1",
            )
        }
    )

    _assert_public_sanitization_error(
        invalid_replay,
        public_snapshot,
        public_state,
        "Official-victory outcomes require a finished game state.",
    )


def test_training_truncation_is_not_recorded_as_a_game_victory() -> None:
    state = _prepared_state()
    snapshot = create_snapshot(
        "training-truncation-initial",
        state,
        visibility="full",
    )
    outcome = TrainingTruncationOutcome(
        outcome_type="training_truncation",
        limit_type="action_count",
        limit_value=500,
    )
    replay = create_full_replay(
        "training-truncation-replay",
        snapshot,
        (),
        state,
        outcome=outcome,
    )

    result = replay_full_replay(
        replay,
        apply_action=lambda current, _action, _streams: current,
        apply_chance_outcome=lambda current, _outcome, _streams: current,
    )

    assert result.outcome == outcome
    assert result.state.is_finished is False
    assert result.state.winner_player_id is None
    with pytest.raises(
        ValueError,
        match="Official-victory outcomes require a finished game state",
    ):
        create_full_replay(
            "false-victory-replay",
            snapshot,
            (),
            state,
            outcome=OfficialVictoryOutcome(
                outcome_type="official_victory",
                winner_player_id="player-1",
            ),
        )


def test_full_replay_creation_rejects_mismatched_game_ids() -> None:
    state = _prepared_state()
    snapshot = create_snapshot(
        "full-replay-mismatched-game",
        state,
        visibility="full",
    )
    final_state = state.model_copy(update={"game_id": "another-game"})

    with pytest.raises(ValueError) as error_info:
        create_full_replay(
            "mismatched-game-replay",
            snapshot,
            (),
            final_state,
        )

    assert str(error_info.value) == (
        "Initial snapshot and final state game IDs must match."
    )


def test_full_replay_creation_rejects_finished_state_without_winner() -> None:
    initial_state = _prepared_state()
    snapshot = create_snapshot(
        "full-replay-finished-without-winner",
        initial_state,
        visibility="full",
    )
    final_state = initial_state.model_copy(
        update={
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "is_finished": True,
            "winner_player_id": None,
        }
    )

    with pytest.raises(ValueError) as error_info:
        create_full_replay(
            "finished-without-winner-replay",
            snapshot,
            (),
            final_state,
        )

    assert str(error_info.value) == "Finished games require an official winner."


def test_individual_elimination_replay_end_does_not_finish_the_game() -> None:
    state, eliminated_player_id = _state_with_individual_elimination()
    snapshot = create_snapshot(
        "individual-elimination-initial",
        state,
        visibility="full",
    )
    outcome = IndividualEliminationOutcome(
        outcome_type="individual_elimination",
        eliminated_player_id=eliminated_player_id,
    )

    replay = create_full_replay(
        "individual-elimination-replay",
        snapshot,
        (),
        state,
        outcome=outcome,
    )

    assert replay.outcome == outcome
    assert isinstance(replay.outcome, IndividualEliminationOutcome)
    assert state.is_finished is False
    assert state.winner_player_id is None


def test_finished_replay_records_the_official_winner() -> None:
    state_data = _prepared_state().model_dump(mode="python")
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
    final_state = FullGameState.model_validate(state_data)
    snapshot = create_snapshot(
        "official-victory-initial",
        final_state,
        visibility="full",
    )

    replay = create_full_replay(
        "official-victory-replay",
        snapshot,
        (),
        final_state,
    )
    result = replay_full_replay(
        replay,
        apply_action=lambda current, _action, _streams: current,
        apply_chance_outcome=lambda current, _outcome, _streams: current,
    )

    assert replay.outcome == OfficialVictoryOutcome(
        outcome_type="official_victory",
        winner_player_id="player-1",
    )
    assert result.outcome == replay.outcome


@pytest.mark.parametrize(
    "outcome",
    (
        ResignationOutcome(
            outcome_type="resignation",
            player_id="player-1",
        ),
        OperationalInterruptionOutcome(
            outcome_type="operational_interruption",
            cause_code="service_unavailable",
        ),
    ),
)
def test_nonvictory_replay_endings_do_not_assign_a_winner(
    outcome: ReplayOutcome,
) -> None:
    state = _prepared_state()
    snapshot = create_snapshot(
        "nonvictory-ending-initial",
        state,
        visibility="full",
    )

    replay = create_full_replay(
        "nonvictory-ending-replay",
        snapshot,
        (),
        state,
        outcome=outcome,
    )

    assert replay.outcome == outcome
    assert state.is_finished is False
    assert state.winner_player_id is None


@pytest.mark.parametrize(
    ("request_ids", "event_ids", "duplicate_label"),
    (
        (
            ("request-1", "request-2"),
            ("event-1", "event-1"),
            "Chance event IDs",
        ),
        (
            ("request-1", "request-1"),
            ("event-1", "event-2"),
            "Chance request IDs",
        ),
    ),
)
def test_replay_rejects_reused_chance_outcome_ids(
    request_ids: tuple[str, str],
    event_ids: tuple[str, str],
    duplicate_label: str,
) -> None:
    initial_state = _prepared_state()
    initial_snapshot = create_snapshot(
        "duplicate-chance-initial",
        initial_state,
        visibility="full",
    )
    entries = tuple(
        ReplayChanceEntry(
            entry_type="chance_outcome",
            sequence_number=index,
            outcome=ChanceOutcome(
                contract_type="chance_outcome",
                request_id=request_id,
                event_id=event_id,
                outcome=[index],
                visibility="public",
                outcome_source="site",
            ),
        )
        for index, (request_id, event_id) in enumerate(
            zip(request_ids, event_ids, strict=True),
            start=1,
        )
    )

    with pytest.raises(ValidationError, match=duplicate_label):
        create_full_replay(
            "duplicate-chance-replay",
            initial_snapshot,
            entries,
            initial_state,
        )


def test_replay_rejects_incompatible_versions_and_divergence() -> None:
    initial_state = _prepared_state()
    snapshot = create_snapshot("incompat-initial", initial_state, visibility="full")
    action = legal_actions(initial_state)[0]
    final_state = apply_action(initial_state, action)
    replay = create_full_replay(
        "incompat-replay",
        snapshot,
        (
            ReplayActionEntry(
                entry_type="action",
                sequence_number=1,
                action=action,
            ),
        ),
        final_state,
    )
    incompatible_replay = replay.model_copy(update={"replay_version": "2.0.0"})

    with pytest.raises(IncompatibleReplaySchemaError, match="Unsupported replay"):
        replay_full_replay(
            incompatible_replay,
            apply_action=lambda state, _action, _streams: state,
            apply_chance_outcome=lambda state, _outcome, _streams: state,
        )
    with pytest.raises(ReplayDivergenceError, match="final state hash"):
        replay_full_replay(
            replay,
            apply_action=lambda state, _action, _streams: state,
            apply_chance_outcome=lambda state, _outcome, _streams: state,
        )


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_snapshot_restore_and_replay_preserve_generated_placement(
    seed: int,
) -> None:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )
    initial_state = prepare_game("metamorphic-replay", players, seed).state
    initial_snapshot = create_snapshot(
        "metamorphic-replay-initial",
        initial_state,
        visibility="full",
    )
    restored = restore_snapshot(initial_snapshot)
    assert isinstance(restored.state, FullGameState)
    assert restored.state == initial_state
    action = legal_actions(initial_state)[0]
    final_state = apply_action(initial_state, action)
    assert apply_action(restored.state, action) == final_state

    replay = create_full_replay(
        "metamorphic-replay",
        initial_snapshot,
        (
            ReplayActionEntry(
                entry_type="action",
                sequence_number=1,
                action=action,
            ),
        ),
        final_state,
    )

    def replay_action(
        current_state: FullGameState,
        replay_action: Action,
        _streams: RandomStreams,
    ) -> FullGameState:
        return apply_action(current_state, replay_action)

    def unexpected_chance(
        current_state: FullGameState,
        _outcome: ChanceOutcome,
        _streams: RandomStreams,
    ) -> FullGameState:
        raise AssertionError("Placement replay must not contain chance outcomes.")

    result = replay_full_replay(
        replay,
        apply_action=replay_action,
        apply_chance_outcome=unexpected_chance,
    )

    assert result.state == final_state
    assert result.random_streams is not None
    assert result.random_streams.snapshot() == final_state.random_streams_state
    assert result.outcome is None
