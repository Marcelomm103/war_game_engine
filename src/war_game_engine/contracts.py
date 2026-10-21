from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .versioning import (
    ACTION_SCHEMA_VERSION,
    API_VERSION,
    CONTRACT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    ActionSchemaVersionLiteral,
    ApiVersionLiteral,
    CertificateManifest,
    ContractSchemaVersionLiteral,
    ObservationSchemaVersionLiteral,
    SemanticVersion,
    SemanticVersionVector,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IncompatibleContractVersionError(ValueError):
    def __init__(
        self, dimension: str, actual_version: object, expected_version: str
    ) -> None:
        self.dimension = dimension
        self.actual_version = actual_version
        self.expected_version = expected_version
        super().__init__(
            f"Unsupported {dimension} version {actual_version!r}; "
            f"expected {expected_version!r}."
        )


class TerritoryDefinition(ContractModel):
    territory_id: str
    canonical_name: str
    continent_id: str
    aliases: tuple[str, ...] = ()


class ContinentDefinition(ContractModel):
    continent_id: str
    name: str
    reinforcement_bonus: NonNegativeInt


class Adjacency(ContractModel):
    first_territory_id: str
    second_territory_id: str


class MapSpec(ContractModel):
    map_id: str
    version: SemanticVersion
    territories: tuple[TerritoryDefinition, ...]
    continents: tuple[ContinentDefinition, ...]
    adjacencies: tuple[Adjacency, ...]


class DeckSpec(ContractModel):
    deck_id: str
    version: str
    card_ids: tuple[str, ...]


class ObjectiveDefinition(ContractModel):
    objective_id: str
    description: str
    certified_rule_ids: tuple[str, ...]


class RulesetSpec(ContractModel):
    contract_type: Literal["ruleset_spec"]
    ruleset_id: str
    edition_id: str
    rules_version: SemanticVersion
    map: MapSpec
    decks: tuple[DeckSpec, ...]
    objectives: tuple[ObjectiveDefinition, ...]
    phases: tuple[str, ...]
    formulas: dict[str, str]
    certified_rule_ids: tuple[str, ...]


class Point2D(ContractModel):
    x: float
    y: float


class TerritoryGeometry(ContractModel):
    territory_id: str
    center: Point2D
    polygon: tuple[Point2D, ...] = ()
    label_anchor: Point2D | None = None


class BoardGeometry(ContractModel):
    contract_type: Literal["board_geometry"]
    geometry_version: SemanticVersion
    coordinate_space: Literal["pixels", "normalized"]
    canvas_width: PositiveInt
    canvas_height: PositiveInt
    territories: tuple[TerritoryGeometry, ...]


class TerritoryState(ContractModel):
    territory_id: str
    owner_player_id: str | None
    army_count: NonNegativeInt


class PublicPlayerState(ContractModel):
    player_id: str
    color: str
    territory_count: NonNegativeInt
    army_count: NonNegativeInt
    is_eliminated: bool


class PlayerPrivateState(ContractModel):
    player_id: str
    objective_id: str | None
    hand_card_ids: tuple[str, ...]


class FullPlayerState(ContractModel):
    public: PublicPlayerState
    private: PlayerPrivateState

    @model_validator(mode="after")
    def private_state_matches_player(self) -> FullPlayerState:
        if self.public.player_id != self.private.player_id:
            raise ValueError("Public and private player IDs must match.")
        return self


class DeckState(ContractModel):
    draw_pile_card_ids: tuple[str, ...]
    discard_pile_card_ids: tuple[str, ...]
    trade_count: NonNegativeInt


class FullGameState(ContractModel):
    contract_type: Literal["full_game_state"]
    game_id: str
    ruleset_id: str
    rules_version: SemanticVersion
    state_version: NonNegativeInt
    phase_id: str
    turn_number: NonNegativeInt
    active_player_id: str | None
    players: tuple[FullPlayerState, ...]
    territories: tuple[TerritoryState, ...]
    deck: DeckState
    engine_random_state: JsonValue
    is_finished: bool


class PublicGameState(ContractModel):
    contract_type: Literal["public_game_state"]
    game_id: str
    ruleset_id: str
    rules_version: SemanticVersion
    state_version: NonNegativeInt
    phase_id: str
    turn_number: NonNegativeInt
    active_player_id: str | None
    players: tuple[PublicPlayerState, ...]
    territories: tuple[TerritoryState, ...]
    is_finished: bool


class ArmyPlacement(ContractModel):
    territory_id: str
    army_count: PositiveInt


class ClaimTerritoryOperation(ContractModel):
    action_type: Literal["claim_territory"]
    territory_id: str


class PlaceArmiesOperation(ContractModel):
    action_type: Literal["place_armies"]
    placements: tuple[ArmyPlacement, ...] = Field(min_length=1)
    placement_phase: Literal["initial", "reinforcement"]


class TradeCardsOperation(ContractModel):
    action_type: Literal["trade_cards"]
    card_ids: tuple[str, str, str]


class AttackOperation(ContractModel):
    action_type: Literal["attack"]
    origin_territory_id: str
    target_territory_id: str
    dice_count: Annotated[int, Field(ge=1, le=3)]


class OccupyTerritoryOperation(ContractModel):
    action_type: Literal["occupy_territory"]
    origin_territory_id: str
    target_territory_id: str
    army_count: PositiveInt


class MoveArmiesOperation(ContractModel):
    action_type: Literal["move_armies"]
    origin_territory_id: str
    target_territory_id: str
    army_count: PositiveInt


class EndTurnOperation(ContractModel):
    action_type: Literal["end_turn"]


ActionOperation = Annotated[
    ClaimTerritoryOperation
    | PlaceArmiesOperation
    | TradeCardsOperation
    | AttackOperation
    | OccupyTerritoryOperation
    | MoveArmiesOperation
    | EndTurnOperation,
    Field(discriminator="action_type"),
]


class Action(ContractModel):
    contract_type: Literal["action"]
    action_id: str
    actor_player_id: str
    action_schema_version: ActionSchemaVersionLiteral
    expected_state_version: NonNegativeInt
    operation: ActionOperation


class GameEvent(ContractModel):
    contract_type: Literal["game_event"]
    event_id: str
    game_id: str
    sequence_number: NonNegativeInt
    state_version: NonNegativeInt
    rules_version: SemanticVersion
    event_type: str
    actor_player_id: str | None
    visibility: Literal["public", "private", "engine"]
    visible_to_player_ids: tuple[str, ...] = ()
    occurred_at: datetime
    data: dict[str, JsonValue]
    state_hash: Sha256Hash | None = None


class PlayerObservation(ContractModel):
    contract_type: Literal["player_observation"]
    game_id: str
    player_id: str
    observation_schema_version: ObservationSchemaVersionLiteral
    state_version: NonNegativeInt
    public_state: PublicGameState
    private_state: PlayerPrivateState
    legal_actions: tuple[Action, ...]
    recent_events: tuple[GameEvent, ...] = ()

    @model_validator(mode="after")
    def validate_player_scope(self) -> PlayerObservation:
        if self.public_state.game_id != self.game_id:
            raise ValueError("Observation and public state game IDs must match.")
        if self.public_state.state_version != self.state_version:
            raise ValueError("Observation and public state versions must match.")
        if self.private_state.player_id != self.player_id:
            raise ValueError("Player observation may contain only its own private state.")
        for action in self.legal_actions:
            if action.actor_player_id != self.player_id:
                raise ValueError("Legal actions must belong to the observed player.")
            if action.expected_state_version != self.state_version:
                raise ValueError("Legal actions must target the observed state version.")
        for event in self.recent_events:
            if event.visibility == "engine":
                raise ValueError("Engine-private events cannot enter a player observation.")
            if (
                event.visibility == "private"
                and self.player_id not in event.visible_to_player_ids
            ):
                raise ValueError("Observation contains an event private to another player.")
        return self


class KnownFact(ContractModel):
    fact_id: str
    subject_id: str
    predicate: str
    value: JsonValue
    source_event_id: str | None
    known_at: datetime


class PlayerKnowledgeState(ContractModel):
    contract_type: Literal["player_knowledge_state"]
    game_id: str
    player_id: str
    state_version: NonNegativeInt
    known_facts: tuple[KnownFact, ...]


class BeliefHypothesis(ContractModel):
    hypothesis_id: str
    probability: Confidence
    assignments: dict[str, JsonValue]


class BeliefState(ContractModel):
    contract_type: Literal["belief_state"]
    game_id: str
    player_id: str
    state_version: NonNegativeInt
    conditioned_on_event_ids: tuple[str, ...]
    hypotheses: tuple[BeliefHypothesis, ...]


T = TypeVar("T")


class ObservedValue(ContractModel, Generic[T]):
    status: Literal["known", "unknown", "ambiguous"]
    value: T | None = None
    confidence: Confidence | None = None
    alternatives: tuple[T, ...] = ()

    @model_validator(mode="after")
    def validate_completeness(self) -> ObservedValue[T]:
        if self.status == "known":
            if self.value is None or self.confidence is None or self.alternatives:
                raise ValueError(
                    "Known observations require a value and confidence only."
                )
        elif self.status == "unknown":
            if (
                self.value is not None
                or self.confidence is not None
                or self.alternatives
            ):
                raise ValueError("Unknown observations cannot invent a value.")
        elif self.value is not None or len(self.alternatives) < 2:
            raise ValueError(
                "Ambiguous observations require at least two candidate values."
            )
        return self


class ImageRegion(ContractModel):
    region_id: str
    frame_id: str
    x: NonNegativeInt
    y: NonNegativeInt
    width: PositiveInt
    height: PositiveInt


class ObservedTerritory(ContractModel):
    territory_id: str
    owner: ObservedValue[str]
    army_count: ObservedValue[NonNegativeInt]
    evidence_region_ids: tuple[str, ...]


class ObservedBoard(ContractModel):
    contract_type: Literal["observed_board"]
    observation_id: str
    captured_at: datetime
    source: Literal["camera", "site", "manual"]
    geometry_version: SemanticVersion
    state_version: NonNegativeInt | None
    territories: tuple[ObservedTerritory, ...]
    image_regions: tuple[ImageRegion, ...]


class ChanceRequest(ContractModel):
    contract_type: Literal["chance_request"]
    request_id: str
    game_id: str
    request_type: Literal["combat_dice", "draw_card", "shuffle_deck"]
    requested_by_player_id: str | None
    parameters: dict[str, JsonValue]


class ChanceOutcome(ContractModel):
    contract_type: Literal["chance_outcome"]
    request_id: str
    event_id: str
    outcome: JsonValue
    visibility: Literal["public", "private"]
    visible_to_player_ids: tuple[str, ...] = ()


SnapshotState = Annotated[
    PublicGameState | FullGameState | PlayerObservation,
    Field(discriminator="contract_type"),
]


class Snapshot(ContractModel):
    contract_type: Literal["snapshot"]
    snapshot_id: str
    game_id: str
    state_version: NonNegativeInt
    captured_at: datetime
    visibility: Literal["public", "full", "player_private"]
    subject_player_id: str | None
    state: SnapshotState
    state_hash: Sha256Hash

    @model_validator(mode="after")
    def state_matches_visibility(self) -> Snapshot:
        valid_state = {
            "public": PublicGameState,
            "full": FullGameState,
            "player_private": PlayerObservation,
        }[self.visibility]
        if not isinstance(self.state, valid_state):
            raise ValueError("Snapshot state does not match its visibility.")
        if (
            self.visibility == "player_private"
            and isinstance(self.state, PlayerObservation)
            and self.subject_player_id != self.state.player_id
        ):
            raise ValueError("Private snapshot subject must match its player.")
        return self


class AgentProtocol(ContractModel):
    contract_type: Literal["agent_protocol"]
    protocol_version: SemanticVersion
    agent_id: str
    supported_ruleset_ids: tuple[str, ...]
    observation_schema_version: SemanticVersion
    observation_encoding_version: SemanticVersion
    action_schema_version: SemanticVersion
    action_encoding_version: SemanticVersion
    memory_scope: Literal["per_player"]
    rng_scope: Literal["private_per_player"]
    decision_budget_ms: PositiveInt


class ActionIntent(ContractModel):
    contract_type: Literal["action_intent"]
    intent_id: str
    expected_state_version: NonNegativeInt
    action: Action
    source: Literal["agent", "touch", "voice", "site", "manual"]
    created_at: datetime

    @model_validator(mode="after")
    def action_targets_expected_state(self) -> ActionIntent:
        if self.action.expected_state_version != self.expected_state_version:
            raise ValueError("Intent and action state versions must match.")
        return self


class ExecutionReceipt(ContractModel):
    contract_type: Literal["execution_receipt"]
    intent_id: str
    status: Literal["executed", "rejected", "uncertain"]
    state_version_before: NonNegativeInt
    state_version_after: NonNegativeInt | None
    event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    error_code: str | None


class Ack(ContractModel):
    contract_type: Literal["ack"]
    intent_id: str
    status: Literal["confirmed", "rejected", "needs_confirmation"]
    acknowledged_at: datetime
    evidence_ids: tuple[str, ...]
    message: str | None


class RulesetVersionReference(ContractModel):
    ruleset_id: str
    rules_version: SemanticVersion


class DatasetVersionReference(ContractModel):
    dataset_id: str
    dataset_version: SemanticVersion


class PolicyManifest(ContractModel):
    contract_type: Literal["policy_manifest"]
    policy_id: str
    policy_version: SemanticVersion
    model_version: SemanticVersion
    weights_sha256: Sha256Hash
    compatible_rulesets: tuple[RulesetVersionReference, ...]
    observation_schema_version: SemanticVersion
    observation_encoding_version: SemanticVersion
    action_schema_version: SemanticVersion
    action_encoding_version: SemanticVersion
    replay_versions: tuple[SemanticVersion, ...]
    dataset_versions: tuple[DatasetVersionReference, ...]
    inference_config_version: SemanticVersion
    search_configuration: dict[str, JsonValue]
    sampling_configuration: dict[str, JsonValue]


class ReplayManifest(ContractModel):
    contract_type: Literal["replay_manifest"]
    replay_id: str
    replay_version: SemanticVersion
    api_version: SemanticVersion
    contract_schema_version: str
    ruleset_id: str
    rules_version: SemanticVersion
    action_schema_version: SemanticVersion
    event_count: NonNegativeInt
    artifact_sha256: Sha256Hash


class DatasetManifest(ContractModel):
    contract_type: Literal["dataset_manifest"]
    dataset_id: str
    dataset_version: SemanticVersion
    ruleset_id: str
    rules_version: SemanticVersion
    observation_schema_version: SemanticVersion
    observation_encoding_version: SemanticVersion
    action_schema_version: SemanticVersion
    action_encoding_version: SemanticVersion
    source_replay_versions: tuple[SemanticVersion, ...]
    record_count: NonNegativeInt
    artifact_sha256: Sha256Hash


class InferenceConfig(ContractModel):
    contract_type: Literal["inference_config"]
    inference_config_version: SemanticVersion
    model_version: SemanticVersion
    decision_budget_ms: PositiveInt
    search_configuration: dict[str, JsonValue]
    sampling_configuration: dict[str, JsonValue]


ContractPayload = Annotated[
    RulesetSpec
    | FullGameState
    | PublicGameState
    | PlayerObservation
    | ObservedBoard
    | BoardGeometry
    | PlayerKnowledgeState
    | BeliefState
    | Action
    | ChanceRequest
    | ChanceOutcome
    | GameEvent
    | Snapshot
    | AgentProtocol
    | ActionIntent
    | ExecutionReceipt
    | Ack
    | PolicyManifest
    | ReplayManifest
    | DatasetManifest
    | InferenceConfig
    | SemanticVersionVector
    | CertificateManifest,
    Field(discriminator="contract_type"),
]


class ContractEnvelope(ContractModel):
    schema_version: ContractSchemaVersionLiteral
    api_version: ApiVersionLiteral
    payload: ContractPayload


def _validate_action_schema_version(value: object) -> None:
    if not isinstance(value, dict):
        return
    actual_version = value.get("action_schema_version")
    if actual_version != ACTION_SCHEMA_VERSION:
        raise IncompatibleContractVersionError(
            "action schema", actual_version, ACTION_SCHEMA_VERSION
        )


def _validate_live_payload_versions(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    contract_type = payload.get("contract_type")
    if contract_type == "action":
        _validate_action_schema_version(payload)
    elif contract_type == "player_observation":
        actual_version = payload.get("observation_schema_version")
        if actual_version != OBSERVATION_SCHEMA_VERSION:
            raise IncompatibleContractVersionError(
                "observation schema",
                actual_version,
                OBSERVATION_SCHEMA_VERSION,
            )
        legal_actions = payload.get("legal_actions")
        if isinstance(legal_actions, list):
            for action in legal_actions:
                _validate_action_schema_version(action)
    elif contract_type == "action_intent":
        _validate_action_schema_version(payload.get("action"))
    elif contract_type == "snapshot":
        _validate_live_payload_versions(payload.get("state"))


def parse_contract_json(serialized: str | bytes) -> ContractEnvelope:
    decoded: object = json.loads(serialized)
    if not isinstance(decoded, dict):
        raise ValueError("Contract envelope must be a JSON object.")
    actual_schema_version = decoded.get("schema_version")
    if actual_schema_version != CONTRACT_SCHEMA_VERSION:
        raise IncompatibleContractVersionError(
            "contract schema", actual_schema_version, CONTRACT_SCHEMA_VERSION
        )
    actual_api_version = decoded.get("api_version")
    if actual_api_version != API_VERSION:
        raise IncompatibleContractVersionError(
            "API", actual_api_version, API_VERSION
        )
    _validate_live_payload_versions(decoded.get("payload"))
    return ContractEnvelope.model_validate_json(serialized)


def contract_json_schema() -> dict[str, object]:
    schema: dict[str, object] = ContractEnvelope.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"urn:war:contracts:{CONTRACT_SCHEMA_VERSION}"
    return schema
