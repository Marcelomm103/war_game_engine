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
    expected_distributor_player_id: str
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
        distributor_player_id = raw_scenario.get("expected_distributor_player_id")
        colors = raw_scenario.get("colors")
        eligible_count = raw_scenario.get("expected_eligible_objectives")
        territory_counts = raw_scenario.get("expected_sorted_territory_counts")
        if (
            not isinstance(name, str)
            or not isinstance(seed, int)
            or not isinstance(distributor_player_id, str)
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
                expected_distributor_player_id=distributor_player_id,
                players=player_specs,
                expected_eligible_objectives=eligible_count,
                expected_sorted_territory_counts=tuple(territory_counts),
            )
        )
    return tuple(scenarios)


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda item: item.name)
@pytest.mark.rule_id(
    "W1-PLAYER-COUNT-01",
    "W1-COLOR-SET-01",
    "W1-OBJECTIVE-DRAW-01",
    "W1-OBJECTIVE-FILTER-01",
    "W1-TERRITORY-DEAL-01",
    "W1-TERRITORY-DEAL-SHUFFLE-01",
    "W1-FIRST-REINFORCEMENT-01",
    "W1-INITIAL-REINFORCEMENT-01",
    "W1-INITIAL-PLACEMENT-01",
    "W1-INITIAL-CONTINENT-BONUS-01",
    "W1-ROUND-ORDER-01",
    "W1-REINFORCEMENT-TERR-01",
    "W1-REINFORCEMENT-CONT-01",
)
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
    assert result.distributor_player_id == scenario.expected_distributor_player_id
    assert tuple(
        player_id for player_id, _ in result.distributor_roll_rounds[0]
    ) == tuple(player.player_id for player in scenario.players)
    assert len(result.territory_deal_order) == len(result.territory_recipients) == 42

    seat_order = tuple(player.player_id for player in scenario.players)
    distributor_index = seat_order.index(scenario.expected_distributor_player_id)
    expected_distribution_order = tuple(
        seat_order[(distributor_index + offset) % len(seat_order)]
        for offset in range(1, len(seat_order) + 1)
    )
    expected_recipients = tuple(
        expected_distribution_order[index % len(expected_distribution_order)]
        for index in range(42)
    )
    assert result.territory_recipients == expected_recipients

    expected_last_recipient = seat_order[
        (distributor_index + len(result.territory_deal_order)) % len(seat_order)
    ]
    assert result.territory_recipients[-1] == expected_last_recipient
    expected_initial_order = tuple(
        seat_order[
            (seat_order.index(expected_last_recipient) + offset) % len(seat_order)
        ]
        for offset in range(1, len(seat_order) + 1)
    )
    assert state.initial_placement_order == expected_initial_order

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


@pytest.mark.rule_id("W1-OBJECTIVE-DRAW-01", "W1-TERRITORY-DEAL-SHUFFLE-01")
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


@pytest.mark.rule_id("W1-OBJECTIVE-FILTER-01", "W1-OBJECTIVE-DRAW-01")
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


@pytest.mark.rule_id("W1-OBJECTIVE-FALLBACK-01")
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


@pytest.mark.rule_id("W1-DISTRIBUTOR-01")
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


@pytest.mark.rule_id("W1-DISTRIBUTOR-01")
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
                SetupPlayer(player_id="p2", color="blue"),
                SetupPlayer(player_id="p3", color="green"),
                SetupPlayer(player_id="p4", color="white"),
                SetupPlayer(player_id="p5", color="black"),
                SetupPlayer(player_id="p6", color="yellow"),
                SetupPlayer(player_id="p7", color="red"),
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
                SetupPlayer(player_id=" ", color="red"),
                SetupPlayer(player_id="p2", color="blue"),
                SetupPlayer(player_id="p3", color="green"),
            ),
            "Player IDs must not be blank",
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
@pytest.mark.rule_id("W1-PLAYER-COUNT-01", "W1-COLOR-SET-01")
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


@pytest.mark.rule_id("W1-DISTRIBUTOR-01")
def test_invalid_distributor_roll_is_an_error() -> None:
    with pytest.raises(InvalidSetupError, match="between 1 and 6"):
        _choose_distributor(("p1",), lambda _player_id: 0)


@pytest.mark.rule_id("W1-DISTRIBUTOR-01")
def test_boolean_distributor_roll_is_not_a_valid_die_face() -> None:
    with pytest.raises(InvalidSetupError, match="between 1 and 6"):
        _choose_distributor(("p1",), lambda _player_id: True)


def test_prepare_game_rejects_incomplete_board_catalogs() -> None:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )
    incomplete_map = WAR1_BOARD_SPEC.map.model_copy(
        update={"territories": WAR1_BOARD_SPEC.map.territories[:-1]}
    )
    incomplete_board = WAR1_BOARD_SPEC.model_copy(update={"map": incomplete_map})
    with pytest.raises(InvalidSetupError, match="42 territories and 14 objectives"):
        prepare_game("incomplete-map", players, 1, board=incomplete_board)

    missing_objectives = tuple(
        objective.model_copy(update={"objective_id": "W1-OBJ-99"})
        if objective.objective_id == "W1-OBJ-01"
        else objective
        for objective in WAR1_BOARD_SPEC.objectives
    )
    incomplete_objectives = WAR1_BOARD_SPEC.model_copy(
        update={"objectives": missing_objectives}
    )
    with pytest.raises(InvalidSetupError, match="missing setup objectives"):
        prepare_game(
            "incomplete-objectives",
            players,
            2,
            board=incomplete_objectives,
        )

    filtered_objectives = tuple(
        objective.model_copy(update={"objective_id": "W1-OBJ-03"})
        if objective.objective_id == "W1-OBJ-14"
        else objective
        for objective in WAR1_BOARD_SPEC.objectives
    )
    invalid_filter_board = WAR1_BOARD_SPEC.model_copy(
        update={"objectives": filtered_objectives}
    )
    with pytest.raises(InvalidSetupError, match="unexpected number"):
        prepare_game(
            "invalid-objective-filter",
            players,
            3,
            board=invalid_filter_board,
        )

    incomplete_deck = WAR1_BOARD_SPEC.territory_deck.model_copy(
        update={
            "card_ids": tuple(
                card_id
                for card_id in WAR1_BOARD_SPEC.territory_deck.card_ids
                if card_id != "T42"
            )
        }
    )
    invalid_deck_board = WAR1_BOARD_SPEC.model_copy(
        update={"territory_deck": incomplete_deck}
    )
    with pytest.raises(
        InvalidSetupError,
        match="one card for each canonical territory",
    ):
        prepare_game(
            "incomplete-territory-deck",
            players,
            4,
            board=invalid_deck_board,
        )


@pytest.mark.rule_id("W1-INITIAL-CONTINENT-BONUS-01", "W1-CONTINENT-BONUS-01")
def test_initial_setup_applies_a_complete_continent_bonus() -> None:
    players = (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    )

    result = prepare_game("initial-continent-bonus", players, 0)

    oceania_territory_ids = ("T01", "T02", "T03", "T04")
    owners = {
        territory.territory_id: territory.owner_player_id
        for territory in result.state.territories
    }
    assert {owners[territory_id] for territory_id in oceania_territory_ids} == {
        "player-1"
    }
    bonus_pools = tuple(
        pool
        for pool in result.state.reinforcement_pools
        if pool.player_id == "player-1" and pool.origin == "continent_bonus"
    )
    assert len(bonus_pools) == 1
    assert bonus_pools[0].remaining_armies == 2
    assert bonus_pools[0].allowed_territory_ids == oceania_territory_ids
    assert result.state.phase_id == "initial_placement"
    assert result.state.turn_number == 1
