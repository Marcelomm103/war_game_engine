"""Canonical snapshots and visibility-safe replay construction and execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypeAlias

from .contracts import (
    Action,
    ChanceOutcome,
    FullGameState,
    FullReplay,
    PlayerObservation,
    PublicGameState,
    PublicReplay,
    ReplayActionEntry,
    ReplayChanceEntry,
    ReplayEntry,
    Snapshot,
)
from .randomness import RandomStreams
from .serialization import canonical_json_bytes, canonical_sha256
from .state import to_public_game_state
from .versioning import API_VERSION, CONTRACT_SCHEMA_VERSION, REPLAY_VERSION

SnapshotState: TypeAlias = FullGameState | PublicGameState | PlayerObservation
FullActionHandler: TypeAlias = Callable[
    [FullGameState, Action, RandomStreams], FullGameState
]
FullChanceHandler: TypeAlias = Callable[
    [FullGameState, ChanceOutcome, RandomStreams], FullGameState
]
PublicActionHandler: TypeAlias = Callable[[PublicGameState, Action], PublicGameState]
PublicChanceHandler: TypeAlias = Callable[
    [PublicGameState, ChanceOutcome], PublicGameState
]


class SnapshotRestoreError(ValueError):
    """Raised when a snapshot cannot be restored without loss."""


class IncompatibleSnapshotSchemaError(SnapshotRestoreError):
    """Raised when a snapshot uses an unsupported replay schema version."""


class IncompatibleReplaySchemaError(ValueError):
    """Raised when a replay uses incompatible contract or replay versions."""


class ReplayDivergenceError(ValueError):
    """Raised when replayed events do not produce the recorded final state."""


@dataclass(frozen=True)
class RestoredSnapshot:
    state: SnapshotState
    random_streams: RandomStreams | None


@dataclass(frozen=True)
class ReplayExecutionResult:
    state: FullGameState | PublicGameState
    random_streams: RandomStreams | None


def create_snapshot(
    snapshot_id: str,
    state: SnapshotState,
    *,
    visibility: Literal["public", "full", "player_private"],
    subject_player_id: str | None = None,
    captured_at: datetime | None = None,
) -> Snapshot:
    if visibility == "full":
        if not isinstance(state, FullGameState):
            raise SnapshotRestoreError("Full snapshots require FullGameState.")
        if state.random_streams_state is None:
            raise SnapshotRestoreError(
                "Full snapshots require a restorable random stream state."
            )
    elif visibility == "public":
        if not isinstance(state, PublicGameState):
            raise SnapshotRestoreError("Public snapshots require PublicGameState.")
        if subject_player_id is not None:
            raise SnapshotRestoreError("Public snapshots cannot have a subject player.")
    elif visibility == "player_private":
        if not isinstance(state, PlayerObservation):
            raise SnapshotRestoreError(
                "Player-private snapshots require PlayerObservation."
            )
        if subject_player_id != state.player_id:
            raise SnapshotRestoreError(
                "Private snapshot subject must match its player."
            )
    else:
        raise SnapshotRestoreError(f"Unsupported snapshot visibility {visibility!r}.")

    return Snapshot(
        contract_type="snapshot",
        snapshot_id=snapshot_id,
        replay_version=REPLAY_VERSION,
        game_id=state.game_id,
        state_version=state.state_version,
        captured_at=captured_at or datetime.now(timezone.utc),
        visibility=visibility,
        subject_player_id=subject_player_id,
        state=state,
        state_hash=canonical_sha256(state),
    )


def restore_snapshot(
    snapshot: Snapshot,
    *,
    expected_replay_version: str = REPLAY_VERSION,
) -> RestoredSnapshot:
    if snapshot.replay_version != expected_replay_version:
        raise IncompatibleSnapshotSchemaError(
            f"Unsupported snapshot replay version {snapshot.replay_version!r}; "
            f"expected {expected_replay_version!r}."
        )
    actual_hash = canonical_sha256(snapshot.state)
    if actual_hash != snapshot.state_hash:
        raise SnapshotRestoreError("Snapshot state hash no longer matches its state.")

    serialized_state = canonical_json_bytes(snapshot.state)
    if isinstance(snapshot.state, FullGameState):
        restored_full_state = FullGameState.model_validate_json(serialized_state)
        if restored_full_state.random_streams_state is None:
            raise SnapshotRestoreError(
                "Full snapshot does not include restorable random streams."
            )
        random_streams = RandomStreams.restore(restored_full_state.random_streams_state)
        return RestoredSnapshot(
            state=restored_full_state,
            random_streams=random_streams,
        )
    if isinstance(snapshot.state, PublicGameState):
        restored_public_state = PublicGameState.model_validate_json(serialized_state)
        return RestoredSnapshot(state=restored_public_state, random_streams=None)
    restored_observation = PlayerObservation.model_validate_json(serialized_state)
    return RestoredSnapshot(state=restored_observation, random_streams=None)


def create_full_replay(
    replay_id: str,
    initial_snapshot: Snapshot,
    entries: tuple[ReplayEntry, ...],
    final_state: FullGameState,
) -> FullReplay:
    if initial_snapshot.game_id != final_state.game_id:
        raise ValueError("Initial snapshot and final state game IDs must match.")
    return FullReplay(
        contract_type="full_replay",
        replay_id=replay_id,
        game_id=final_state.game_id,
        replay_version=REPLAY_VERSION,
        api_version=API_VERSION,
        contract_schema_version=CONTRACT_SCHEMA_VERSION,
        ruleset_id=final_state.ruleset_id,
        rules_version=final_state.rules_version,
        initial_snapshot=initial_snapshot,
        entries=entries,
        final_state_hash=canonical_sha256(final_state),
    )


def sanitize_public_replay(
    full_replay: FullReplay,
    public_replay_id: str,
    public_initial_snapshot: Snapshot,
    public_final_state: PublicGameState,
) -> PublicReplay:
    if full_replay.initial_snapshot.visibility != "full":
        raise ValueError("Only a full replay can be sanitized.")
    full_initial_state = full_replay.initial_snapshot.state
    if not isinstance(full_initial_state, FullGameState):
        raise ValueError("Full replay initial snapshot must contain FullGameState.")
    if public_initial_snapshot.visibility != "public":
        raise ValueError("Sanitized replays require a public initial snapshot.")
    public_initial_state = public_initial_snapshot.state
    if not isinstance(public_initial_state, PublicGameState):
        raise ValueError("Public replay snapshot must contain PublicGameState.")
    if (
        public_initial_snapshot.game_id != full_replay.game_id
        or public_final_state.game_id != full_replay.game_id
    ):
        raise ValueError("Public snapshots must match the full replay game ID.")
    if to_public_game_state(full_initial_state) != public_initial_state:
        raise ValueError("Public initial snapshot does not match the full snapshot.")
    if (
        public_final_state.game_id != full_initial_state.game_id
        or public_final_state.ruleset_id != full_replay.ruleset_id
        or public_final_state.rules_version != full_replay.rules_version
    ):
        raise ValueError("Public final state does not match the full replay.")

    public_entries: list[ReplayEntry] = []
    for entry in full_replay.entries:
        if isinstance(entry, ReplayActionEntry):
            public_entries.append(
                ReplayActionEntry(
                    entry_type="action",
                    sequence_number=len(public_entries) + 1,
                    action=entry.action,
                )
            )
        elif entry.outcome.visibility == "public":
            public_entries.append(
                ReplayChanceEntry(
                    entry_type="chance_outcome",
                    sequence_number=len(public_entries) + 1,
                    outcome=ChanceOutcome(
                        contract_type="chance_outcome",
                        request_id=entry.outcome.request_id,
                        event_id=entry.outcome.event_id,
                        outcome=entry.outcome.outcome,
                        visibility="public",
                    ),
                )
            )

    return PublicReplay(
        contract_type="public_replay",
        replay_id=public_replay_id,
        game_id=full_replay.game_id,
        replay_version=full_replay.replay_version,
        api_version=full_replay.api_version,
        contract_schema_version=full_replay.contract_schema_version,
        ruleset_id=full_replay.ruleset_id,
        rules_version=full_replay.rules_version,
        initial_snapshot=public_initial_snapshot,
        entries=tuple(public_entries),
        final_state_hash=canonical_sha256(public_final_state),
    )


def _validate_replay_versions(
    replay_version: str,
    api_version: str,
    contract_schema_version: str,
) -> None:
    if replay_version != REPLAY_VERSION:
        raise IncompatibleReplaySchemaError(
            f"Unsupported replay version {replay_version!r}; "
            f"expected {REPLAY_VERSION!r}."
        )
    if api_version != API_VERSION:
        raise IncompatibleReplaySchemaError(
            f"Unsupported replay API version {api_version!r}; expected {API_VERSION!r}."
        )
    if contract_schema_version != CONTRACT_SCHEMA_VERSION:
        raise IncompatibleReplaySchemaError(
            f"Unsupported replay contract schema version "
            f"{contract_schema_version!r}; expected {CONTRACT_SCHEMA_VERSION!r}."
        )


def replay_full_replay(
    replay: FullReplay,
    *,
    apply_action: FullActionHandler,
    apply_chance_outcome: FullChanceHandler,
) -> ReplayExecutionResult:
    _validate_replay_versions(
        replay.replay_version,
        replay.api_version,
        replay.contract_schema_version,
    )
    restored = restore_snapshot(replay.initial_snapshot)
    if not isinstance(restored.state, FullGameState) or restored.random_streams is None:
        raise SnapshotRestoreError("Full replay could not restore full state and RNG.")
    state = restored.state
    streams = restored.random_streams
    for entry in replay.entries:
        if isinstance(entry, ReplayActionEntry):
            next_state = apply_action(state, entry.action, streams)
        else:
            next_state = apply_chance_outcome(state, entry.outcome, streams)
        if (
            next_state.game_id != replay.game_id
            or next_state.ruleset_id != replay.ruleset_id
            or next_state.rules_version != replay.rules_version
        ):
            raise ReplayDivergenceError(
                "Replay reducer returned a state for another game or ruleset."
            )
        state = next_state
        state_data = state.model_dump(mode="python")
        state_data["engine_random_state"] = None
        state_data["random_streams_state"] = streams.snapshot()
        state = FullGameState.model_validate(state_data)
    if canonical_sha256(state) != replay.final_state_hash:
        raise ReplayDivergenceError(
            "Full replay did not reproduce the recorded final state hash."
        )
    return ReplayExecutionResult(state=state, random_streams=streams)


def replay_public_replay(
    replay: PublicReplay,
    *,
    apply_action: PublicActionHandler,
    apply_chance_outcome: PublicChanceHandler,
) -> ReplayExecutionResult:
    _validate_replay_versions(
        replay.replay_version,
        replay.api_version,
        replay.contract_schema_version,
    )
    restored = restore_snapshot(replay.initial_snapshot)
    if not isinstance(restored.state, PublicGameState):
        raise SnapshotRestoreError("Public replay could not restore public state.")
    state = restored.state
    for entry in replay.entries:
        if isinstance(entry, ReplayActionEntry):
            next_state = apply_action(state, entry.action)
        else:
            next_state = apply_chance_outcome(state, entry.outcome)
        if (
            next_state.game_id != replay.game_id
            or next_state.ruleset_id != replay.ruleset_id
            or next_state.rules_version != replay.rules_version
        ):
            raise ReplayDivergenceError(
                "Replay reducer returned a state for another game or ruleset."
            )
        state = next_state
    if canonical_sha256(state) != replay.final_state_hash:
        raise ReplayDivergenceError(
            "Public replay did not reproduce the recorded final state hash."
        )
    return ReplayExecutionResult(state=state, random_streams=None)


__all__ = [
    "IncompatibleReplaySchemaError",
    "IncompatibleSnapshotSchemaError",
    "ReplayDivergenceError",
    "ReplayExecutionResult",
    "RestoredSnapshot",
    "SnapshotRestoreError",
    "create_full_replay",
    "create_snapshot",
    "replay_full_replay",
    "replay_public_replay",
    "restore_snapshot",
    "sanitize_public_replay",
]
