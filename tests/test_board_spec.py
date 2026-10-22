import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from war_game_engine import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    BoardSpec,
    ContractEnvelope,
    DeckSpec,
    MapSpec,
    ObjectiveDefinition,
    RulesetSpec,
    parse_contract_json,
)

ROOT = Path(__file__).parents[3]


def _valid_map_data() -> dict[str, object]:
    return {
        "map_id": "test-map",
        "version": "1.0.0",
        "territories": [
            {
                "territory_id": "A",
                "canonical_name": "Alfa",
                "continent_id": "main",
            },
            {
                "territory_id": "B",
                "canonical_name": "Beta",
                "continent_id": "main",
            },
        ],
        "continents": [
            {
                "continent_id": "main",
                "name": "Main",
                "reinforcement_bonus": 1,
            }
        ],
        "adjacencies": [{"first_territory_id": "A", "second_territory_id": "B"}],
    }


def _validate_map(data: dict[str, object]) -> MapSpec:
    return MapSpec.model_validate_json(json.dumps(data))


def _validate_board(data: dict[str, object]) -> BoardSpec:
    return BoardSpec.model_validate_json(json.dumps(data))


def test_canonical_war1_board_has_expected_composition() -> None:
    board = WAR1_BOARD_SPEC
    territories = board.map.territories
    continent_counts = Counter(territory.continent_id for territory in territories)
    symbols = Counter(territory.card_symbol for territory in territories)

    assert len(territories) == 42
    assert continent_counts == {
        "north_america": 9,
        "south_america": 4,
        "europe": 7,
        "africa": 6,
        "asia": 12,
        "oceania": 4,
    }
    assert symbols == {"triangle": 14, "circle": 14, "square": 14}
    assert len(board.map.adjacencies) == 79
    assert len(board.territory_deck.card_ids) == 44
    assert len(board.objectives) == len(board.objective_deck.card_ids) == 14

    bonuses = {
        continent.continent_id: continent.reinforcement_bonus
        for continent in board.map.continents
    }
    assert bonuses == {
        "north_america": 5,
        "south_america": 2,
        "europe": 5,
        "africa": 3,
        "asia": 7,
        "oceania": 2,
    }

    colombia = next(t for t in territories if t.territory_id == "T27")
    bolivia = next(t for t in territories if t.territory_id == "T42")
    assert (colombia.canonical_name, colombia.aliases) == ("Colômbia", ("Venezuela",))
    assert (bolivia.canonical_name, bolivia.aliases) == (
        "Bolívia",
        ("Peru", "Chile"),
    )
    assert not {territory.canonical_name for territory in territories}.intersection(
        {"Venezuela", "Peru", "Chile"}
    )


def test_canonical_adjacencies_match_the_user_corrected_json() -> None:
    source_path = ROOT / "doc" / "war_adjacency_candidate_v1.0.json"
    source: object = json.loads(source_path.read_text(encoding="utf-8"))
    assert isinstance(source, dict)
    source_adjacencies = source.get("adjacency")
    assert isinstance(source_adjacencies, dict)

    source_edges: set[frozenset[str]] = set()
    for territory_id, neighbors in source_adjacencies.items():
        assert isinstance(territory_id, str)
        assert isinstance(neighbors, list)
        for neighbor in neighbors:
            assert isinstance(neighbor, str)
            source_edges.add(frozenset((territory_id, neighbor)))

    canonical_edges = {
        frozenset((adjacency.first_territory_id, adjacency.second_territory_id))
        for adjacency in WAR1_BOARD_SPEC.map.adjacencies
    }
    assert len(source_edges) == 79
    assert canonical_edges == source_edges


def test_board_spec_round_trips_through_contract_envelope() -> None:
    envelope = ContractEnvelope(
        schema_version="1.0",
        api_version="1.0.0",
        payload=WAR1_BOARD_SPEC,
    )

    parsed = parse_contract_json(envelope.model_dump_json())

    assert parsed.payload == WAR1_BOARD_SPEC


def test_every_canonical_entry_has_provenance_and_asset_hashes_match() -> None:
    board = WAR1_BOARD_SPEC
    references = [
        *board.source_references,
        *board.map.source_references,
        *board.territory_deck.source_references,
        *board.objective_deck.source_references,
    ]
    for territory in board.map.territories:
        assert territory.source_references
        references.extend(territory.source_references)
    for continent in board.map.continents:
        assert continent.source_references
        references.extend(continent.source_references)
    for adjacency in board.map.adjacencies:
        assert adjacency.source_references
        references.extend(adjacency.source_references)
    for objective in board.objectives:
        assert objective.source_references
        references.extend(objective.source_references)

    for reference in references:
        if reference.sha256 is None:
            continue
        asset_path = ROOT / reference.source
        actual_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        assert reference.sha256 == f"sha256:{actual_hash}"


def test_map_rejects_duplicate_territory_ids() -> None:
    data = _valid_map_data()
    territories = data["territories"]
    assert isinstance(territories, list)
    duplicate = dict(territories[0])
    duplicate["canonical_name"] = "Gamma"
    territories.append(duplicate)

    with pytest.raises(ValidationError, match="Territory IDs must be unique"):
        _validate_map(data)


def test_map_rejects_duplicate_continent_ids() -> None:
    data = _valid_map_data()
    continents = data["continents"]
    assert isinstance(continents, list)
    duplicate = dict(continents[0])
    duplicate["name"] = "Other"
    continents.append(duplicate)

    with pytest.raises(ValidationError, match="Continent IDs must be unique"):
        _validate_map(data)


def test_map_rejects_duplicate_names_and_aliases_case_and_accent_insensitively() -> (
    None
):
    data = _valid_map_data()
    territories = data["territories"]
    assert isinstance(territories, list)
    territories[1]["aliases"] = (" ÁLFA ",)

    with pytest.raises(
        ValidationError, match="Territory names and aliases must be unique"
    ):
        _validate_map(data)


def test_map_rejects_unknown_continents_and_empty_continents() -> None:
    unknown_data = _valid_map_data()
    unknown_territories = unknown_data["territories"]
    assert isinstance(unknown_territories, list)
    unknown_territories[1]["continent_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown continent IDs"):
        _validate_map(unknown_data)

    empty_data = _valid_map_data()
    empty_continents = empty_data["continents"]
    assert isinstance(empty_continents, list)
    empty_continents.append(
        {
            "continent_id": "empty",
            "name": "Empty",
            "reinforcement_bonus": 0,
        }
    )
    with pytest.raises(ValidationError, match="Continents without territories"):
        _validate_map(empty_data)


@pytest.mark.parametrize(
    ("adjacencies", "message"),
    (
        (
            [
                {"first_territory_id": "A", "second_territory_id": "missing"},
            ],
            "unknown territory IDs",
        ),
        (
            [
                {"first_territory_id": "A", "second_territory_id": "A"},
            ],
            "cannot be adjacent to itself",
        ),
        (
            [
                {"first_territory_id": "A", "second_territory_id": "B"},
                {"first_territory_id": "B", "second_territory_id": "A"},
            ],
            "Duplicate adjacency",
        ),
    ),
)
def test_map_rejects_invalid_adjacencies(
    adjacencies: list[dict[str, str]], message: str
) -> None:
    data = _valid_map_data()
    data["adjacencies"] = adjacencies

    with pytest.raises(ValidationError, match=message):
        _validate_map(data)


def test_map_rejects_disconnected_graphs() -> None:
    data = _valid_map_data()
    data["territories"] = [
        {"territory_id": "A", "canonical_name": "Alfa", "continent_id": "main"},
        {"territory_id": "B", "canonical_name": "Beta", "continent_id": "main"},
        {"territory_id": "C", "canonical_name": "Gamma", "continent_id": "main"},
        {"territory_id": "D", "canonical_name": "Delta", "continent_id": "main"},
    ]
    data["adjacencies"] = [
        {"first_territory_id": "A", "second_territory_id": "B"},
        {"first_territory_id": "C", "second_territory_id": "D"},
    ]

    with pytest.raises(ValidationError, match="Map graph is disconnected"):
        _validate_map(data)


def test_deck_rejects_duplicate_card_ids() -> None:
    with pytest.raises(ValidationError, match="Card IDs must be unique"):
        DeckSpec(
            deck_id="test-deck",
            version="1.0.0",
            card_ids=("same", "same"),
        )


def test_board_rejects_duplicate_and_unmatched_objectives() -> None:
    duplicate_id_data = WAR1_BOARD_SPEC.model_dump(mode="json")
    duplicate_id_data["objectives"][1]["objective_id"] = "W1-OBJ-01"
    with pytest.raises(ValidationError, match="Objective IDs must be unique"):
        _validate_board(duplicate_id_data)

    duplicate_description_data = WAR1_BOARD_SPEC.model_dump(mode="json")
    duplicate_description_data["objectives"][1]["description"] = (
        duplicate_description_data["objectives"][0]["description"]
    )
    with pytest.raises(ValidationError, match="Objective descriptions must be unique"):
        _validate_board(duplicate_description_data)

    missing_card_data = WAR1_BOARD_SPEC.model_dump(mode="json")
    missing_card_data["objective_deck"]["card_ids"].pop()
    with pytest.raises(
        ValidationError, match="Objective deck must contain exactly one card"
    ):
        _validate_board(missing_card_data)


def test_board_rejects_incomplete_territory_deck() -> None:
    data = WAR1_BOARD_SPEC.model_dump(mode="json")
    data["territory_deck"]["card_ids"].remove("T42")

    with pytest.raises(
        ValidationError, match="Territory deck must contain exactly one card"
    ):
        _validate_board(data)


def test_board_requires_provenance_for_every_canonical_datum() -> None:
    data = WAR1_BOARD_SPEC.model_dump(mode="json")
    data["map"]["territories"][0]["source_references"] = []

    with pytest.raises(ValidationError, match="must include source references"):
        _validate_board(data)


def test_ruleset_rejects_duplicate_deck_ids_and_unknown_objective_rules() -> None:
    board = WAR1_BOARD_SPEC
    deck = DeckSpec(deck_id="shared", version="1.0.0", card_ids=("card-1",))
    objective = ObjectiveDefinition(
        objective_id="objective-1",
        description="Unique objective",
        certified_rule_ids=("rule-1",),
    )
    base = {
        "contract_type": "ruleset_spec",
        "ruleset_id": "test-ruleset",
        "edition_id": "test-edition",
        "rules_version": "1.0.0",
        "map": board.map,
        "decks": (deck,),
        "objectives": (objective,),
        "phases": ("setup",),
        "formulas": {},
        "certified_rule_ids": ("rule-1",),
    }

    with pytest.raises(ValidationError, match="Deck IDs must be unique"):
        RulesetSpec.model_validate({**base, "decks": (deck, deck)})
    with pytest.raises(ValidationError, match="references unknown certified rule IDs"):
        RulesetSpec.model_validate({**base, "certified_rule_ids": ("other-rule",)})
