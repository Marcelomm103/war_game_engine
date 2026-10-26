import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from war_game_engine.contracts import (
    BoardGeometry,
    ContractEnvelope,
    IncompatibleContractVersionError,
    PlayerObservation,
    PublicGameState,
    TerritoryState,
    contract_json_schema,
    parse_contract_json,
)
from war_game_engine.versioning import (
    API_VERSION,
    CONTRACT_SCHEMA_VERSION,
)

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


@pytest.mark.parametrize(
    "fixture_name",
    (
        "public_game_state.json",
        "player_observation_private.json",
        "observed_board_incomplete.json",
    ),
)
def test_documented_examples_validate_and_round_trip(fixture_name: str) -> None:
    serialized = (FIXTURES / fixture_name).read_text(encoding="utf-8")

    envelope = parse_contract_json(serialized)
    round_tripped = parse_contract_json(envelope.model_dump_json())

    assert round_tripped == envelope


def test_incompatible_schema_version_is_rejected_explicitly() -> None:
    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported contract schema version '2.0'",
    ):
        parse_contract_json(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "payload": {"contract_type": "public_game_state"},
                }
            )
        )


def test_unknown_schema_version_is_rejected_explicitly() -> None:
    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported contract schema version None",
    ):
        parse_contract_json(
            json.dumps(
                {
                    "payload": {"contract_type": "public_game_state"},
                }
            )
        )


def test_incompatible_api_version_is_rejected_explicitly() -> None:
    serialized = (FIXTURES / "public_game_state.json").read_text(encoding="utf-8")
    data = json.loads(serialized)
    data["api_version"] = "2.0.0"

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported API version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_incompatible_observation_schema_version_is_rejected_explicitly() -> None:
    serialized = (FIXTURES / "player_observation_private.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(serialized)
    data["payload"]["observation_schema_version"] = "2.0.0"

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported observation schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_incompatible_action_schema_version_is_rejected_explicitly() -> None:
    serialized = (FIXTURES / "player_observation_private.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(serialized)
    data["payload"]["legal_actions"][0]["action_schema_version"] = "2.0.0"

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported action schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_action_intent_action_version_is_prevalidated_explicitly() -> None:
    data = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "payload": {
            "contract_type": "action_intent",
            "intent_id": "intent-1",
            "expected_state_version": 0,
            "action": {"action_schema_version": "2.0.0"},
        },
    }

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported action schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_action_payload_version_is_prevalidated_explicitly() -> None:
    data = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "payload": {
            "contract_type": "action",
            "action_schema_version": "2.0.0",
        },
    }

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported action schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_snapshot_payload_prevalidates_nested_live_contract_versions() -> None:
    data = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "payload": {
            "contract_type": "snapshot",
            "state": {
                "contract_type": "action",
                "action_schema_version": "2.0.0",
            },
        },
    }

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported action schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


@pytest.mark.parametrize(
    ("contract_type", "payload_state", "entries"),
    (
        (
            "full_replay",
            {
                "contract_type": "action",
                "action_schema_version": "2.0.0",
            },
            [],
        ),
        (
            "public_replay",
            {"contract_type": "public_game_state"},
            [
                {
                    "entry_type": "action",
                    "action": {"action_schema_version": "2.0.0"},
                }
            ],
        ),
    ),
)
def test_replay_payload_prevalidates_nested_action_versions(
    contract_type: str,
    payload_state: dict[str, object],
    entries: list[dict[str, object]],
) -> None:
    data = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "payload": {
            "contract_type": contract_type,
            "initial_snapshot": {"state": payload_state},
            "entries": entries,
        },
    }

    with pytest.raises(
        IncompatibleContractVersionError,
        match="Unsupported action schema version '2.0.0'",
    ):
        parse_contract_json(json.dumps(data))


def test_public_game_state_rejects_private_player_fields() -> None:
    serialized = (FIXTURES / "public_game_state.json").read_text(encoding="utf-8")
    data = json.loads(serialized)
    data["payload"]["players"][0]["objective_id"] = "secret-objective"

    with pytest.raises(ValidationError):
        parse_contract_json(json.dumps(data))


def test_private_observation_cannot_contain_another_players_private_state() -> None:
    serialized = (FIXTURES / "player_observation_private.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(serialized)
    data["payload"]["private_state"]["player_id"] = "player-2"

    with pytest.raises(ValidationError, match="only its own private state"):
        parse_contract_json(json.dumps(data))


def test_logical_territory_state_does_not_contain_visual_geometry() -> None:
    assert "center" not in TerritoryState.model_fields
    assert "polygon" not in TerritoryState.model_fields
    assert "owner_player_id" not in BoardGeometry.model_fields
    assert "army_count" not in BoardGeometry.model_fields


def test_generated_schema_matches_models() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "war_game_engine"
        / "schemas"
        / "contract-envelope-v1.json"
    )
    generated = json.loads(schema_path.read_text(encoding="utf-8"))

    assert generated == contract_json_schema()


def test_player_observation_rejects_mismatched_public_state_version() -> None:
    serialized = (FIXTURES / "player_observation_private.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(serialized)
    data["payload"]["public_state"]["state_version"] = 2

    with pytest.raises(ValidationError, match="versions must match"):
        parse_contract_json(json.dumps(data))


def test_contract_envelope_requires_version_one() -> None:
    assert CONTRACT_SCHEMA_VERSION == "1.0"
    assert ContractEnvelope.model_fields["schema_version"].is_required()
    assert ContractEnvelope.model_fields["api_version"].is_required()
    assert PlayerObservation.model_fields["player_id"].is_required()
    assert PublicGameState.model_fields["state_version"].is_required()
