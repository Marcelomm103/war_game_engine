"""Run a reproducible random-policy stress campaign against the WAR engine."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    ChanceOutcome,
    DrawTerritoryCardOperation,
    FullGameState,
    ReplayActionEntry,
    ReplayChanceEntry,
    ReplayEntry,
    Snapshot,
)
from war_game_engine.engine import (
    apply_action,
    apply_combat_outcome,
    legal_actions,
    simulate_pending_combat_outcome,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.replay import (
    ReplayDivergenceError,
    create_full_replay,
    create_snapshot,
    replay_full_replay,
)
from war_game_engine.setup import PlayerColor, SetupPlayer, prepare_game

PLAYER_COLORS: tuple[PlayerColor, ...] = (
    "white",
    "red",
    "black",
    "blue",
    "yellow",
    "green",
)
MAX_RECORDED_ISSUES = 20


@dataclass
class StressReport:
    requested_games: int
    max_transitions_per_game: int
    seed: int
    workers: int = 1
    attempted_games: int = 0
    completed_games: int = 0
    truncated_games: int = 0
    crashed_games: int = 0
    transitions: int = 0
    invariant_checks: int = 0
    invariant_violations: int = 0
    replay_checks: int = 0
    replay_divergences: int = 0
    phase_transition_counts: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    issues: list[str] = field(default_factory=list)


def _record_issue(report: StressReport, issue: str) -> None:
    if len(report.issues) < MAX_RECORDED_ISSUES:
        report.issues.append(issue)


def _with_random_streams(
    state: FullGameState,
    streams: RandomStreams,
) -> FullGameState:
    state_data = state.model_dump(mode="python")
    state_data["random_streams_state"] = streams.snapshot()
    return FullGameState.model_validate(state_data)


def _validate_stress_state(state: FullGameState) -> FullGameState:
    return FullGameState.model_validate(state.model_dump(mode="python"))


def _simulate_one_game(
    game_index: int,
    report: StressReport,
) -> tuple[Snapshot, FullGameState, tuple[ReplayEntry, ...]] | None:
    seed = report.seed + game_index
    player_count = 3 + game_index % 4
    players = tuple(
        SetupPlayer(
            player_id=f"player-{index + 1}",
            color=PLAYER_COLORS[index],
        )
        for index in range(player_count)
    )
    try:
        initial_state = prepare_game(
            f"stress-{game_index}",
            players,
            seed,
        ).state
        initial_snapshot = create_snapshot(
            f"stress-snapshot-{game_index}",
            initial_state,
            visibility="full",
        )
        state = _validate_stress_state(initial_state)
    except ValidationError as error:
        report.invariant_violations += 1
        report.crashed_games += 1
        _record_issue(
            report,
            f"game {game_index}: initial-state invariant failure: {error}",
        )
        return None
    except Exception as error:
        report.crashed_games += 1
        _record_issue(
            report,
            f"game {game_index}: initialization crash "
            f"{type(error).__name__}: {error}",
        )
        return None

    entries: list[ReplayEntry] = []
    game_transitions = 0
    while not state.is_finished and game_transitions < report.max_transitions_per_game:
        source_phase = state.phase_id
        try:
            if state.pending_combat is not None:
                outcome, random_streams_state = simulate_pending_combat_outcome(state)
                state = apply_combat_outcome(
                    state,
                    outcome,
                    random_streams_state=random_streams_state,
                    board=WAR1_BOARD_SPEC,
                )
                entries.append(
                    ReplayChanceEntry(
                        entry_type="chance_outcome",
                        sequence_number=len(entries) + 1,
                        outcome=outcome,
                    )
                )
            else:
                actions = legal_actions(state, board=WAR1_BOARD_SPEC)
                if not actions:
                    raise RuntimeError(
                        f"No legal actions in nonterminal phase {state.phase_id!r}."
                    )
                decision_player_id = (
                    state.decision_player_id or state.active_player_id
                )
                if decision_player_id is None:
                    raise RuntimeError(
                        "A nonterminal stress state has no decision player."
                    )
                if state.random_streams_state is None:
                    raise RuntimeError(
                        "A stress state has no restorable random stream state."
                    )
                streams = RandomStreams.restore(state.random_streams_state)
                policy_rng = streams.policy_generator(
                    f"stress:{decision_player_id}"
                )
                action_index = int(policy_rng.integers(0, len(actions)))
                action = actions[action_index]
                state = _with_random_streams(state, streams)
                state = apply_action(state, action, board=WAR1_BOARD_SPEC)
                entries.append(
                    ReplayActionEntry(
                        entry_type="action",
                        sequence_number=len(entries) + 1,
                        action=action,
                    )
                )

            state = _validate_stress_state(state)
        except ValidationError as error:
            report.invariant_violations += 1
            report.crashed_games += 1
            _record_issue(
                report,
                f"game {game_index}: transition invariant failure: {error}",
            )
            return None
        except Exception as error:
            report.crashed_games += 1
            _record_issue(
                report,
                f"game {game_index}: transition crash "
                f"{type(error).__name__}: {error}",
            )
            return None

        report.transitions += 1
        report.invariant_checks += 1
        report.phase_transition_counts[source_phase] = (
            report.phase_transition_counts.get(source_phase, 0) + 1
        )
        game_transitions += 1

    if state.is_finished:
        report.completed_games += 1
    else:
        report.truncated_games += 1

    return initial_snapshot, state, tuple(entries)


def _replay_action(
    state: FullGameState,
    action: Action,
    streams: RandomStreams,
) -> FullGameState:
    actions = legal_actions(state, board=WAR1_BOARD_SPEC)
    if not actions:
        raise ReplayDivergenceError(
            "Stress replay has no legal action for its recorded action entry."
        )
    policy_rng = streams.policy_generator(f"stress:{action.actor_player_id}")
    expected_action = actions[int(policy_rng.integers(0, len(actions)))]
    if expected_action != action:
        raise ReplayDivergenceError(
            f"Stress replay selected {expected_action.action_id!r}, not "
            f"{action.action_id!r}."
        )
    result = apply_action(
        _with_random_streams(state, streams),
        action,
        board=WAR1_BOARD_SPEC,
    )
    if (
        isinstance(action.operation, DrawTerritoryCardOperation)
        and not state.deck.draw_pile_card_ids
    ):
        streams.shuffle("cards", state.deck.discard_pile_card_ids)
    return result


def _replay_chance(
    state: FullGameState,
    outcome: ChanceOutcome,
    streams: RandomStreams,
) -> FullGameState:
    pending = state.pending_combat
    if outcome.outcome_source == "simulation":
        if pending is None:
            raise ReplayDivergenceError(
                "Stress replay has no pending combat for a simulated outcome."
            )
        defender_player_id = next(
            territory.owner_player_id
            for territory in state.territories
            if territory.territory_id == pending.target_territory_id
        )
        attacker = next(
            player for player in state.players
            if player.public.player_id == pending.attacker_player_id
        )
        defender = next(
            player for player in state.players
            if player.public.player_id == defender_player_id
        )
        combined_hand = (
            *attacker.private.hand_card_ids,
            *defender.private.hand_card_ids,
        )
        for _ in range(
            pending.attacker_dice_count + pending.defender_dice_count
        ):
            streams.die("combat")
        result = apply_combat_outcome(
            state,
            outcome,
            random_streams_state=streams.snapshot(),
            board=WAR1_BOARD_SPEC,
        )
        eliminated_defender = next(
            player
            for player in result.players
            if player.public.player_id == defender_player_id
        )
        excess_cards = max(0, len(combined_hand) - 5)
        if eliminated_defender.public.is_eliminated and not result.is_finished:
            if excess_cards:
                streams.sample("cards", combined_hand, excess_cards)
        return result
    return apply_combat_outcome(state, outcome, board=WAR1_BOARD_SPEC)


def _verify_replay(
    game_index: int,
    initial_snapshot: Snapshot,
    final_state: FullGameState,
    entries: tuple[ReplayEntry, ...],
) -> None:
    replay = create_full_replay(
        f"stress-replay-{game_index}",
        initial_snapshot,
        entries,
        final_state,
    )
    result = replay_full_replay(
        replay,
        apply_action=_replay_action,
        apply_chance_outcome=_replay_chance,
    )
    if result.state != final_state:
        raise ReplayDivergenceError(
            f"Stress replay for game {game_index} produced a different state."
        )


def _run_game(game_index: int, report: StressReport) -> None:
    game_result = _simulate_one_game(game_index, report)
    if game_result is None:
        return
    initial_snapshot, final_state, entries = game_result
    try:
        _verify_replay(
            game_index,
            initial_snapshot,
            final_state,
            entries,
        )
        report.replay_checks += 1
    except ReplayDivergenceError as error:
        report.replay_divergences += 1
        _record_issue(
            report,
            f"game {game_index}: replay divergence: {error}",
        )
    except Exception as error:
        report.replay_divergences += 1
        _record_issue(
            report,
            f"game {game_index}: replay verification failure "
            f"{type(error).__name__}: {error}",
        )


def _run_campaign_chunk(
    start_index: int,
    games: int,
    max_transitions_per_game: int,
    seed: int,
    workers: int,
) -> StressReport:
    report = StressReport(
        requested_games=games,
        max_transitions_per_game=max_transitions_per_game,
        seed=seed,
        workers=workers,
    )
    for game_index in range(start_index, start_index + games):
        report.attempted_games += 1
        _run_game(game_index, report)
    return report


def run_campaign(
    games: int,
    max_transitions_per_game: int,
    seed: int,
    *,
    workers: int = 1,
) -> StressReport:
    if games < 1:
        raise ValueError("The stress campaign requires at least one game.")
    if max_transitions_per_game < 1:
        raise ValueError("The per-game transition limit must be positive.")
    if seed < 0:
        raise ValueError("The stress seed must be non-negative.")
    if workers < 1:
        raise ValueError("The stress campaign requires at least one worker.")

    report = StressReport(
        requested_games=games,
        max_transitions_per_game=max_transitions_per_game,
        seed=seed,
        workers=workers,
    )
    start_time = time.perf_counter()
    chunk_reports: tuple[StressReport, ...]
    if workers == 1:
        chunk_reports = (
            _run_campaign_chunk(
                0,
                games,
                max_transitions_per_game,
                seed,
                workers,
            ),
        )
    else:
        chunk_size = (games + workers - 1) // workers
        chunk_arguments = tuple(
            (
                start_index,
                min(chunk_size, games - start_index),
                max_transitions_per_game,
                seed,
                workers,
            )
            for start_index in range(0, games, chunk_size)
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = tuple(
                executor.submit(_run_campaign_chunk, *args)
                for args in chunk_arguments
            )
            chunk_reports = tuple(future.result() for future in futures)
    for chunk_report in chunk_reports:
        report.attempted_games += chunk_report.attempted_games
        report.completed_games += chunk_report.completed_games
        report.truncated_games += chunk_report.truncated_games
        report.crashed_games += chunk_report.crashed_games
        report.transitions += chunk_report.transitions
        report.invariant_checks += chunk_report.invariant_checks
        report.invariant_violations += chunk_report.invariant_violations
        report.replay_checks += chunk_report.replay_checks
        report.replay_divergences += chunk_report.replay_divergences
        for phase_id, count in chunk_report.phase_transition_counts.items():
            report.phase_transition_counts[phase_id] = (
                report.phase_transition_counts.get(phase_id, 0) + count
            )
        for issue in chunk_report.issues:
            _record_issue(report, issue)
    report.elapsed_seconds = time.perf_counter() - start_time
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-transitions-per-game", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()

    try:
        report = run_campaign(
            arguments.games,
            arguments.max_transitions_per_game,
            arguments.seed,
            workers=arguments.workers,
        )
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(asdict(report), sort_keys=True))
    return int(
        report.crashed_games > 0
        or report.invariant_violations > 0
        or report.replay_divergences > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
