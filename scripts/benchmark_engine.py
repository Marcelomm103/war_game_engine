"""Measure reference-engine throughput, memory, validation, and action generation."""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import platform
import pstats
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Callable

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    EndPhaseOperation,
    FullGameState,
    FullPlayerState,
    PendingOccupation,
    TerritoryState,
)
from war_game_engine.engine import apply_action, legal_actions, validate_action
from war_game_engine.serialization import canonical_json_bytes, canonical_sha256
from war_game_engine.setup import SetupPlayer, prepare_game

ENGINE_ROOT = Path(__file__).resolve().parents[1]
STRESS_SCRIPT = ENGINE_ROOT / "scripts" / "stress_engine.py"
BENCHMARK_SCHEMA_VERSION = "1.0"
PLAYER_COLORS = ("white", "red", "blue")
PROFILE_ITERATIONS = 20
MAX_SETUP_TRANSITIONS = 1_000


class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark workload cannot produce valid measurements."""


def _require_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _summary(samples_ns: list[int]) -> dict[str, float | int]:
    if not samples_ns:
        raise ValueError("A timing summary requires at least one sample.")
    ordered = sorted(samples_ns)
    percentile_95_index = max(0, ceil(0.95 * len(ordered)) - 1)
    average_ns = mean(samples_ns)
    return {
        "iterations": len(samples_ns),
        "mean_us": round(average_ns / 1_000, 3),
        "median_us": round(median(samples_ns) / 1_000, 3),
        "p95_us": round(ordered[percentile_95_index] / 1_000, 3),
        "operations_per_second": round(1_000_000_000 / average_ns, 2),
    }


def _measure_calls(
    operation: Callable[[], object],
    iterations: int,
) -> list[int]:
    for _ in range(min(10, iterations)):
        operation()

    durations_ns: list[int] = []
    for _ in range(iterations):
        started_at = time.perf_counter_ns()
        result = operation()
        durations_ns.append(time.perf_counter_ns() - started_at)
        del result
    return durations_ns


def _profile_action_generation(
    state: FullGameState,
    profile_top: int,
) -> list[dict[str, int | float | str]]:
    if profile_top == 0:
        return []

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(PROFILE_ITERATIONS):
        legal_actions(state, board=WAR1_BOARD_SPEC)
    profiler.disable()

    profile_stats = pstats.Stats(profiler)
    functions: list[dict[str, int | float | str]] = []
    for (filename, line_number, function_name), stats in profile_stats.stats.items():
        primitive_calls, total_calls, total_seconds, cumulative_seconds, _ = stats
        functions.append(
            {
                "function": f"{Path(filename).name}:{line_number}:{function_name}",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_us": round(total_seconds * 1_000_000, 3),
                "cumulative_us": round(cumulative_seconds * 1_000_000, 3),
            }
        )
    functions.sort(
        key=lambda item: float(item["cumulative_us"]),
        reverse=True,
    )
    return functions[:profile_top]


def _find_phase_end_action(state: FullGameState) -> Action:
    for action in legal_actions(state, board=WAR1_BOARD_SPEC):
        if isinstance(action.operation, EndPhaseOperation):
            return action
    raise BenchmarkExecutionError(
        f"No phase-ending action is available in {state.phase_id!r}."
    )


def _apply_first_legal_action(state: FullGameState) -> FullGameState:
    actions = legal_actions(state, board=WAR1_BOARD_SPEC)
    if not actions:
        raise BenchmarkExecutionError(
            f"No legal action is available in {state.phase_id!r}."
        )
    return apply_action(state, actions[0], board=WAR1_BOARD_SPEC)


def _finish_initial_placement(state: FullGameState) -> FullGameState:
    for _ in range(MAX_SETUP_TRANSITIONS):
        if state.phase_id != "initial_placement":
            if state.phase_id != "reinforcement":
                raise BenchmarkExecutionError(
                    "Initial placement did not lead to the reinforcement phase."
                )
            return state
        state = _apply_first_legal_action(state)
    raise BenchmarkExecutionError("Initial placement exceeded its transition limit.")


def _finish_reinforcement_placement(state: FullGameState) -> FullGameState:
    for _ in range(MAX_SETUP_TRANSITIONS):
        if state.phase_id != "reinforcement_placement":
            if state.phase_id != "attack":
                raise BenchmarkExecutionError(
                    "Reinforcement placement did not lead to the attack phase."
                )
            return state
        state = _apply_first_legal_action(state)
    raise BenchmarkExecutionError(
        "Reinforcement placement exceeded its transition limit."
    )


def _replace_player_counts(
    state_data: dict[str, object],
    territories: tuple[TerritoryState, ...],
) -> None:
    territory_counts: dict[str, int] = {}
    army_counts: dict[str, int] = {}
    for territory in territories:
        if territory.owner_player_id is None:
            continue
        territory_counts[territory.owner_player_id] = (
            territory_counts.get(territory.owner_player_id, 0) + 1
        )
        army_counts[territory.owner_player_id] = (
            army_counts.get(territory.owner_player_id, 0) + territory.army_count
        )

    players = state_data.get("players")
    if not isinstance(players, (list, tuple)):
        raise BenchmarkExecutionError("Benchmark state has no player list.")
    updated_players: list[dict[str, object]] = []
    for player in players:
        if isinstance(player, FullPlayerState):
            player_data = player.model_dump(mode="python")
        elif isinstance(player, dict):
            player_data = dict(player)
        else:
            raise BenchmarkExecutionError("Benchmark state has an invalid player.")
        public = player_data.get("public")
        if not isinstance(public, dict):
            raise BenchmarkExecutionError("Benchmark player has no public state.")
        player_id = public.get("player_id")
        if not isinstance(player_id, str):
            raise BenchmarkExecutionError("Benchmark player has no player ID.")
        public.update(
            {
                "territory_count": territory_counts.get(player_id, 0),
                "army_count": army_counts.get(player_id, 0),
                "is_eliminated": territory_counts.get(player_id, 0) == 0,
            }
        )
        updated_players.append(player_data)
    state_data["players"] = tuple(updated_players)


def _territory_claim_state(state: FullGameState) -> FullGameState:
    owner_counts: dict[str, int] = {}
    for territory in state.territories:
        if territory.owner_player_id is not None:
            owner_counts[territory.owner_player_id] = (
                owner_counts.get(territory.owner_player_id, 0) + 1
            )
    unowned_candidate = next(
        (
            territory
            for territory in state.territories
            if territory.owner_player_id is not None
            and owner_counts[territory.owner_player_id] > 1
        ),
        None,
    )
    if unowned_candidate is None:
        raise BenchmarkExecutionError(
            "Cannot construct a territory-claim benchmark state."
        )
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                None
                if territory.territory_id == unowned_candidate.territory_id
                else territory.owner_player_id
            ),
            army_count=(
                0
                if territory.territory_id == unowned_candidate.territory_id
                else territory.army_count
            ),
        )
        for territory in state.territories
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "territory_claim",
            "reinforcement_pools": (),
            "pending_combat": None,
            "pending_occupation": None,
            "pending_objective_check": None,
            "has_conquered_territory_this_turn": False,
            "territories": territories,
        }
    )
    _replace_player_counts(state_data, territories)
    return FullGameState.model_validate(state_data)


def _state_with_hand(
    state: FullGameState,
    hand_size: int,
    *,
    phase_id: str,
    has_conquered: bool,
) -> FullGameState:
    if state.random_streams_state is None:
        raise BenchmarkExecutionError("Benchmark state has no random streams.")
    if not 0 <= hand_size <= 6 or hand_size > len(state.deck.draw_pile_card_ids):
        raise ValueError("Benchmark hand size must be between zero and six.")
    decision_player_id = state.decision_player_id or state.active_player_id
    if decision_player_id is None:
        raise BenchmarkExecutionError("Benchmark state has no decision player.")

    card_ids = state.deck.draw_pile_card_ids[:hand_size]
    players = []
    for player in state.players:
        player_data = player.model_dump(mode="python")
        if player.public.player_id == decision_player_id:
            player_data["private"]["hand_card_ids"] = card_ids
        players.append(player_data)
    state_data = state.model_dump(mode="python")
    deck_data = state_data["deck"]
    if not isinstance(deck_data, dict):
        raise BenchmarkExecutionError("Benchmark state has an invalid deck.")
    deck_data["draw_pile_card_ids"] = state.deck.draw_pile_card_ids[hand_size:]
    state_data["players"] = tuple(players)
    state_data["deck"] = deck_data
    state_data["phase_id"] = phase_id
    state_data["turn_number"] = max(2, state.turn_number)
    state_data["has_conquered_territory_this_turn"] = has_conquered
    if phase_id in {"card_purchase", "card_discard"}:
        state_data["reinforcement_pools"] = ()
    return FullGameState.model_validate(state_data)


def _occupation_state(state: FullGameState) -> FullGameState:
    decision_player_id = state.decision_player_id or state.active_player_id
    if decision_player_id is None:
        raise BenchmarkExecutionError("Benchmark state has no decision player.")
    territory_by_id = {
        territory.territory_id: territory for territory in state.territories
    }
    route: tuple[TerritoryState, TerritoryState] | None = None
    for adjacency in WAR1_BOARD_SPEC.map.adjacencies:
        first = territory_by_id[adjacency.first_territory_id]
        second = territory_by_id[adjacency.second_territory_id]
        if (
            first.owner_player_id == decision_player_id
            and first.army_count > 1
            and second.owner_player_id not in {None, decision_player_id}
        ):
            route = (first, second)
            break
        if (
            second.owner_player_id == decision_player_id
            and second.army_count > 1
            and first.owner_player_id not in {None, decision_player_id}
        ):
            route = (second, first)
            break
    if route is None:
        raise BenchmarkExecutionError(
            "Cannot construct a legal occupation benchmark state."
        )
    origin, target = route
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                None
                if territory.territory_id == target.territory_id
                else territory.owner_player_id
            ),
            army_count=(
                0
                if territory.territory_id == target.territory_id
                else territory.army_count
            ),
        )
        for territory in state.territories
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "occupation",
            "turn_number": max(2, state.turn_number),
            "active_player_id": decision_player_id,
            "turn_player_id": decision_player_id,
            "decision_player_id": decision_player_id,
            "reinforcement_pools": (),
            "pending_combat": None,
            "pending_occupation": PendingOccupation(
                contract_type="pending_occupation",
                attacker_player_id=decision_player_id,
                defender_player_id=target.owner_player_id,
                origin_territory_id=origin.territory_id,
                target_territory_id=target.territory_id,
                maximum_army_count=min(3, origin.army_count - 1),
            ),
            "pending_objective_check": None,
            "territories": territories,
        }
    )
    _replace_player_counts(state_data, territories)
    return FullGameState.model_validate(state_data)


def _phase_states(seed: int) -> dict[str, FullGameState]:
    players = tuple(
        SetupPlayer(player_id=f"player-{index}", color=color)
        for index, color in enumerate(PLAYER_COLORS, start=1)
    )
    initial_state = prepare_game(
        "benchmark-phase-states",
        players,
        seed,
    ).state
    reinforcement_state = _finish_initial_placement(initial_state)
    reinforcement_placement_state = apply_action(
        reinforcement_state,
        _find_phase_end_action(reinforcement_state),
        board=WAR1_BOARD_SPEC,
    )
    attack_state = _finish_reinforcement_placement(reinforcement_placement_state)
    movement_state = apply_action(
        attack_state,
        _find_phase_end_action(attack_state),
        board=WAR1_BOARD_SPEC,
    )
    turn_end_state = apply_action(
        movement_state,
        _find_phase_end_action(movement_state),
        board=WAR1_BOARD_SPEC,
    )
    return {
        "territory_claim_synthetic": _territory_claim_state(initial_state),
        "initial_placement": initial_state,
        "reinforcement": reinforcement_state,
        "reinforcement_with_five_cards_synthetic": _state_with_hand(
            reinforcement_state,
            5,
            phase_id="reinforcement",
            has_conquered=False,
        ),
        "reinforcement_placement": reinforcement_placement_state,
        "attack": attack_state,
        "occupation_synthetic": _occupation_state(attack_state),
        "movement": movement_state,
        "card_purchase_synthetic": _state_with_hand(
            movement_state,
            5,
            phase_id="card_purchase",
            has_conquered=True,
        ),
        "card_discard_synthetic": _state_with_hand(
            movement_state,
            6,
            phase_id="card_discard",
            has_conquered=True,
        ),
        "turn_end": turn_end_state,
    }


def _measure_setup_memory(seed: int) -> dict[str, int | str]:
    gc.collect()
    tracemalloc.start()
    try:
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        state = prepare_game(
            "benchmark-memory",
            tuple(
                SetupPlayer(player_id=f"player-{index}", color=color)
                for index, color in enumerate(PLAYER_COLORS, start=1)
            ),
            seed,
        ).state
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        return {
            "tracked_python_bytes_retained_by_initial_state": max(
                0, current_bytes - baseline_bytes
            ),
            "tracked_python_bytes_peak_during_setup": max(
                0, peak_bytes - baseline_bytes
            ),
            "canonical_state_json_bytes": len(canonical_json_bytes(state)),
            "measurement_scope": (
                "tracemalloc Python allocations; excludes native allocations"
            ),
        }
    finally:
        tracemalloc.stop()


def _measure_deep_clone_memory(state: FullGameState) -> dict[str, int]:
    gc.collect()
    tracemalloc.start()
    try:
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        cloned_state = state.model_copy(deep=True)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        return {
            "tracked_python_bytes_retained_by_deep_clone": max(
                0, current_bytes - baseline_bytes
            ),
            "tracked_python_bytes_peak_during_deep_clone": max(
                0, peak_bytes - baseline_bytes
            ),
            "clone_state_hash_matches": int(
                canonical_sha256(cloned_state) == canonical_sha256(state)
            ),
        }
    finally:
        tracemalloc.stop()


def _measure_action_generation(
    states: dict[str, FullGameState],
    iterations: int,
    profile_top: int,
) -> dict[str, object]:
    measurements: dict[str, object] = {}
    for state_name, state in states.items():
        actions = legal_actions(state, board=WAR1_BOARD_SPEC)
        if not actions:
            raise BenchmarkExecutionError(
                f"Benchmark phase {state_name!r} has no legal actions."
            )
        for action in actions:
            validate_action(state, action, board=WAR1_BOARD_SPEC)
        latency_samples = _measure_calls(
            lambda game_state=state: legal_actions(
                game_state,
                board=WAR1_BOARD_SPEC,
            ),
            iterations,
        )
        measurements[state_name] = {
            "phase_id": state.phase_id,
            "actions_per_call": len(actions),
            "action_signature_sha256": canonical_sha256(
                [action.model_dump(mode="json") for action in actions]
            ),
            "latency": _summary(latency_samples),
            "top_cumulative_profile_functions": _profile_action_generation(
                state,
                profile_top,
            ),
        }
    return measurements


def run_microbenchmarks(
    *,
    iterations: int,
    seed: int,
    profile_top: int = 5,
) -> dict[str, object]:
    """Measure cloning, Python memory, validation, and phase action generation."""
    _require_positive(iterations, "iterations")
    if isinstance(profile_top, bool) or profile_top < 0:
        raise ValueError("profile_top must be a non-negative integer.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    states = _phase_states(seed)
    measured_state = states["attack"]
    state_payload = measured_state.model_dump(mode="python")
    legal_attack_actions = legal_actions(measured_state, board=WAR1_BOARD_SPEC)
    if not legal_attack_actions:
        raise BenchmarkExecutionError("Attack benchmark has no legal actions.")
    action = legal_attack_actions[0]

    clone_measurements = {
        "pydantic_shallow_model_copy": _summary(
            _measure_calls(
                lambda: measured_state.model_copy(deep=False),
                iterations,
            )
        ),
        "pydantic_deep_model_copy": _summary(
            _measure_calls(
                lambda: measured_state.model_copy(deep=True),
                iterations,
            )
        ),
        "state_model_validate_from_pre_dumped_dict": _summary(
            _measure_calls(
                lambda: FullGameState.model_validate(state_payload),
                iterations,
            )
        ),
    }
    validation_measurements = {
        "full_state_validation_from_pre_dumped_dict": _summary(
            _measure_calls(
                lambda: FullGameState.model_validate(state_payload),
                iterations,
            )
        ),
        "single_legal_attack_action": _summary(
            _measure_calls(
                lambda: validate_action(
                    measured_state,
                    action,
                    board=WAR1_BOARD_SPEC,
                ),
                iterations,
            )
        ),
    }
    return {
        "seed": seed,
        "iterations": iterations,
        "cloning": clone_measurements,
        "memory": {
            **_measure_setup_memory(seed),
            **_measure_deep_clone_memory(measured_state),
        },
        "validation": validation_measurements,
        "action_generation_by_phase": _measure_action_generation(
            states,
            iterations,
            profile_top,
        ),
    }


def _number_from_report(report: dict[str, object], key: str) -> float:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkExecutionError(
            f"Stress report field {key!r} is not numeric."
        )
    return float(value)


def run_game_campaign(
    *,
    games: int,
    max_transitions_per_game: int,
    seed: int,
) -> dict[str, object]:
    """Measure seeded random-policy games with invariant and replay checks."""
    _require_positive(games, "games")
    _require_positive(max_transitions_per_game, "max_transitions_per_game")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    command = (
        sys.executable,
        str(STRESS_SCRIPT),
        "--games",
        str(games),
        "--max-transitions-per-game",
        str(max_transitions_per_game),
        "--seed",
        str(seed),
        "--workers",
        "1",
    )
    completed = subprocess.run(
        command,
        cwd=ENGINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkExecutionError(
            "Stress campaign failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkExecutionError(
            f"Stress campaign did not return valid JSON: {error}"
        ) from error
    if not isinstance(raw_report, dict):
        raise BenchmarkExecutionError("Stress campaign report must be an object.")

    elapsed_seconds = _number_from_report(raw_report, "elapsed_seconds")
    if elapsed_seconds <= 0:
        raise BenchmarkExecutionError(
            "Stress campaign duration must be greater than zero."
        )
    attempted_games = int(_number_from_report(raw_report, "attempted_games"))
    completed_games = int(_number_from_report(raw_report, "completed_games"))
    transitions = int(_number_from_report(raw_report, "transitions"))
    return {
        "seed": seed,
        "requested_games": games,
        "max_transitions_per_game": max_transitions_per_game,
        "attempted_games": attempted_games,
        "completed_games": completed_games,
        "truncated_games": int(_number_from_report(raw_report, "truncated_games")),
        "crashed_games": int(_number_from_report(raw_report, "crashed_games")),
        "transitions": transitions,
        "invariant_violations": int(
            _number_from_report(raw_report, "invariant_violations")
        ),
        "replay_checks": int(_number_from_report(raw_report, "replay_checks")),
        "replay_divergences": int(
            _number_from_report(raw_report, "replay_divergences")
        ),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "transitions_per_second": round(transitions / elapsed_seconds, 2),
        "completed_games_per_second": round(
            completed_games / elapsed_seconds,
            4,
        ),
        "attempted_games_per_second": round(
            attempted_games / elapsed_seconds,
            4,
        ),
        "timed_scope": (
            "engine setup, random legal actions, state invariant validation, "
            "and full replay verification; excludes process startup"
        ),
    }


def _environment_metadata() -> dict[str, str | int | None]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
    }


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-transitions-per-game", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--profile-top", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON report to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": _environment_metadata(),
            "game_campaign": run_game_campaign(
                games=args.games,
                max_transitions_per_game=args.max_transitions_per_game,
                seed=args.seed,
            ),
            "microbenchmarks": run_microbenchmarks(
                iterations=args.iterations,
                seed=args.seed,
                profile_top=args.profile_top,
            ),
        }
    except (BenchmarkExecutionError, OSError, ValueError) as error:
        parser.error(str(error))

    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        _atomic_write(args.output, serialized_report)
    print(serialized_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
