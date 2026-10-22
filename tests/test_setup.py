import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import PlaceArmiesOperation
from war_game_engine.engine import legal_actions
from war_game_engine.setup import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    InvalidSetupError,
    SetupPlayer,
    _choose_distributor,
    _fallback_objective_id,
    prepare_game,
)
from war_game_engine.state import observe_player

FIXTURES = Path(__file__).parent / "fixtures" / "setup"


@dataclass(frozen=True)
class SetupScenario:
    name: str
    seed: int
    players: tuple[SetupPlayer, ...]
    expected_eligible_objectives: int
    expected_sorted_territory_counts: tuple[int, ...]


def _load_scenarios() -> tuple[SetupScenario, ...]:
    source: object = json.loads(
        (FIXTURES / "player_counts.json").read_text(encoding="utf-8")
    )
    if not isinstance(source, dict):
        raise AssertionError("Setup fixture must be a JSON object.")
    raw_scenarios = source.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise AssertionError("Setup fixture must contain a scenarios list.")

    scenarios: list[SetupScenario] = []
    for raw_scenario in raw_scenarios:
        if not isinstance(raw_scenario, dict):
            raise AssertionError("Each setup scenario must be an object.")
        name = raw_scenario.get("name")
        seed = raw_scenario.get("seed")
        colors = raw_scenario.get("colors")
        eligible_count = raw_scenario.get("expected_eligible_objectives")
        territory_counts = raw_scenario.get("expected_sorted_territory_counts")
        if (
            not isinstance(name, str)
            or not isinstance(seed, int)
            or not isinstance(colors, list)
            or not all(isinstance(color, str) for color in colors)
            or not isinstance(eligible_count, int)
            or not isinstance(territory_counts, list)
            or not all(isinstance(count, int) for count in territory_counts)
        ):
            raise AssertionError("Setup fixture scenario fields have invalid types.")
        player_specs = tuple(
            SetupPlayer.model_validate(
                {"player_id": f"player-{index + 1}", "color": color}
            )
            for index, color in enumerate(colors)
        )
        scenarios.append(
            SetupScenario(
                name=name,
                seed=seed,
                players=player_specs,
                expected_eligible_objectives=eligible_count,
                expected_sorted_territory_counts=tuple(territory_counts),
            )
        )
    return tuple(scenarios)


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda item: item.name)
def test_prepare_game_covers_each_supported_player_count(
    scenario: SetupScenario,
) -> None:
    result = prepare_game(
        game_id=f"game-{scenario.name}",
        players=scenario.players,
        seed=scenario.seed,
    )
    state = result.state

    territory_counts = tuple(
        sorted(player.public.territory_count for player in state.players)
    )
    assert territory_counts == scenario.expected_sorted_territory_counts
    assert len(result.eligible_objective_ids) == scenario.expected_eligible_objectives
    assert len(result.objective_draw_order) == len(scenario.players)
    assert len(set(result.objective_draw_order)) == len(scenario.players)
    assert set(result.objective_draw_order).issubset(result.eligible_objective_ids)
    assert len(state.territories) == 42
    assert all(territory.army_count == 1 for territory in state.territories)
    assert len(state.deck.draw_pile_card_ids) == 44
    assert len(set(state.deck.draw_pile_card_ids)) == 44
    assert state.deck.discard_pile_card_ids == ()
    assert state.phase_id == "initial_placement"
    assert state.turn_number == 1
    assert state.turn_player_id == state.decision_player_id
    assert len(state.initial_placement_order) == len(scenario.players)
    assert state.decision_player_id is not None
    assert state.decision_player_id == state.initial_placement_order[0]
    assert tuple(
        player_id for player_id, _ in result.distributor_roll_rounds[0]
    ) == tuple(player.player_id for player in scenario.players)
    assert len(result.territory_deal_order) == len(result.territory_recipients) == 42

    recipient_after_last_card = result.territory_recipients[-1]
    seat_order = tuple(player.player_id for player in scenario.players)
    next_seat = (seat_order.index(recipient_after_last_card) + 1) % len(
        scenario.players
    )
    expected_start = seat_order[next_seat]
    assert state.initial_placement_order[0] == expected_start

    assert {player.private.player_id for player in state.players} == {
        player.player_id for player in scenario.players
    }
    owners = {
        territory.territory_id: territory.owner_player_id
        for territory in state.territories
    }
    for continent in WAR1_BOARD_SPEC.map.continents:
        continent_territory_ids = tuple(
            territory.territory_id
            for territory in WAR1_BOARD_SPEC.map.territories
            if territory.continent_id == continent.continent_id
        )
        for player in state.players:
            pool = next(
                (
                    pool
                    for pool in state.reinforcement_pools
                    if pool.player_id == player.public.player_id
                    and pool.origin == "continent_bonus"
                    and pool.allowed_territory_ids == continent_territory_ids
                ),
                None,
            )
            if all(
                owners[territory_id] == player.public.player_id
                for territory_id in continent_territory_ids
            ):
                assert pool is not None
                assert pool.remaining_armies == continent.reinforcement_bonus
            else:
                assert pool is None
    for player in state.players:
        territory_pool = next(
            pool
            for pool in state.reinforcement_pools
            if pool.player_id == player.public.player_id
            and pool.origin == "territory_count"
        )
        assert territory_pool.remaining_armies == max(
            3, player.public.territory_count // 2
        )

    actions = legal_actions(state)
    assert actions
    assert all(action.actor_player_id == state.decision_player_id for action in actions)
    assert all(isinstance(action.operation, PlaceArmiesOperation) for action in actions)
    assert state.decision_player_id is not None
    observation = observe_player(
        state,
        state.decision_player_id,
        legal_actions=actions,
    )
    serialized_observation = observation.model_dump_json()
    assert "root_seed" not in serialized_observation
    assert "bit_generator_state" not in serialized_observation


def test_setup_is_reproducible_and_uses_separate_random_streams() -> None:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )

    first = prepare_game("same-game", players, 2026)
    repeated = prepare_game("same-game", players, 2026)

    assert first == repeated
    assert first.state.random_streams_state is not None
    assert first.state.engine_random_state is None


def test_objectives_for_absent_colors_are_filtered_without_fallback_defaults() -> None:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )
    result = prepare_game("filtered-objectives", players, 101)

    selected_objectives = {
        player.private.objective_id for player in result.state.players
    }
    eligible_elimination_objectives = {
        objective_id
        for objective_id, target_color in ELIMINATION_OBJECTIVE_COLORS.items()
        if target_color in {player.color for player in players}
    }
    assert len(result.eligible_objective_ids) == 11
    assert selected_objectives.issubset(result.eligible_objective_ids)
    assert not (
        selected_objectives.intersection(ELIMINATION_OBJECTIVE_COLORS)
        - eligible_elimination_objectives
    )
    assert all(
        player.private.objective_id is not None for player in result.state.players
    )


def test_own_color_elimination_mission_uses_the_printed_fallback() -> None:
    assert _fallback_objective_id("W1-OBJ-01", "red") == FALLBACK_OBJECTIVE_ID
    assert _fallback_objective_id("W1-OBJ-03", "red") is None
    assert _fallback_objective_id(FALLBACK_OBJECTIVE_ID, "red") is None

    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )
    fallback_found = False
    for seed in range(50):
        result = prepare_game("fallback-case", players, seed)
        for player in result.state.players:
            if player.private.objective_fallback_id is None:
                continue
            objective_id = player.private.objective_id
            assert objective_id is not None
            assert ELIMINATION_OBJECTIVE_COLORS[objective_id] == player.public.color
            assert player.private.objective_fallback_id == FALLBACK_OBJECTIVE_ID
            fallback_found = True
    assert fallback_found


def test_distributor_ties_reroll_only_the_current_tied_players() -> None:
    outcomes = iter((6, 6, 1, 4, 5))
    roll_order: list[str] = []

    def roll(player_id: str) -> int:
        roll_order.append(player_id)
        return next(outcomes)

    distributor, rounds = _choose_distributor(("p1", "p2", "p3"), roll)

    assert distributor == "p2"
    assert roll_order == ["p1", "p2", "p3", "p1", "p2"]
    assert rounds == (
        (("p1", 6), ("p2", 6), ("p3", 1)),
        (("p1", 4), ("p2", 5)),
    )


def test_persistent_distributor_ties_continue_until_unique_winner() -> None:
    outcomes = iter((4, 4, 4, 6, 6, 2, 3, 5))
    roll_order: list[str] = []

    def roll(player_id: str) -> int:
        roll_order.append(player_id)
        return next(outcomes)

    distributor, rounds = _choose_distributor(("p1", "p2", "p3"), roll)

    assert distributor == "p2"
    assert tuple(len(round_rolls) for round_rolls in rounds) == (3, 3, 2)
    assert roll_order == ["p1", "p2", "p3", "p1", "p2", "p3", "p1", "p2"]


@pytest.mark.parametrize(
    ("players", "message"),
    (
        (
            (
                SetupPlayer(player_id="p1", color="red"),
                SetupPlayer(player_id="p2", color="blue"),
            ),
            "between 3 and 6 players",
        ),
        (
            (
                SetupPlayer(player_id="p1", color="red"),
                SetupPlayer(player_id="p1", color="blue"),
                SetupPlayer(player_id="p3", color="green"),
            ),
            "Player IDs must be unique",
        ),
        (
            (
                SetupPlayer(player_id="p1", color="red"),
                SetupPlayer(player_id="p2", color="red"),
                SetupPlayer(player_id="p3", color="green"),
            ),
            "Player colors must be unique",
        ),
    ),
)
def test_setup_rejects_invalid_player_configuration(
    players: tuple[SetupPlayer, ...], message: str
) -> None:
    with pytest.raises(InvalidSetupError, match=message):
        prepare_game("invalid-game", players, 1)


def test_setup_rejects_invalid_seeds_and_game_ids() -> None:
    players = (
        SetupPlayer(player_id="p1", color="red"),
        SetupPlayer(player_id="p2", color="blue"),
        SetupPlayer(player_id="p3", color="green"),
    )
    with pytest.raises(InvalidSetupError, match="non-negative integer"):
        prepare_game("game", players, True)
    with pytest.raises(InvalidSetupError, match="Game ID"):
        prepare_game(" ", players, 1)


def test_invalid_distributor_roll_is_an_error() -> None:
    with pytest.raises(InvalidSetupError, match="between 1 and 6"):
        _choose_distributor(("p1",), lambda _player_id: 0)
