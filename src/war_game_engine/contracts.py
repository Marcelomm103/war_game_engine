from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .serialization import canonical_sha256
from .versioning import (
    ACTION_SCHEMA_VERSION,
    API_VERSION,
    CONTRACT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REPLAY_VERSION,
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
TerritoryCardSymbol = Literal["triangle", "circle", "square"]
GamePhaseId = Literal[
    "territory_claim",
    "initial_placement",
    "reinforcement",
    "reinforcement_placement",
    "attack",
    "occupation",
    "objective_check",
    "movement",
    "card_purchase",
    "card_discard",
    "turn_end",
    "finished",
]
EndablePhaseId = Literal["reinforcement", "attack", "movement"]
ObjectiveResumePhaseId = Literal[
    "territory_claim",
    "initial_placement",
    "reinforcement",
    "reinforcement_placement",
    "attack",
    "occupation",
    "movement",
    "card_purchase",
    "card_discard",
    "turn_end",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_marks.casefold().split())


def _ensure_unique(
    values: Iterable[str],
    label: str,
    normalizer: Callable[[str], str] | None = None,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = normalizer(value) if normalizer is not None else value
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"{label} must be unique; duplicates: {duplicate_list}.")


def _validate_objectives(objectives: tuple[ObjectiveDefinition, ...]) -> None:
    _ensure_unique(
        (objective.objective_id for objective in objectives), "Objective IDs"
    )
    _ensure_unique(
        (objective.description for objective in objectives),
        "Objective descriptions",
        normalizer=_normalize_text,
    )
    for objective in objectives:
        _ensure_unique(
            objective.certified_rule_ids,
            f"Certified rule IDs for objective {objective.objective_id!r}",
        )


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


class SourceReference(ContractModel):
    source: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    sha256: Sha256Hash | None = None


def _require_source_references(
    source_references: tuple[SourceReference, ...], label: str
) -> None:
    if not source_references:
        raise ValueError(f"{label} must include source references.")


class TerritoryDefinition(ContractModel):
    territory_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    continent_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    card_symbol: TerritoryCardSymbol | None = None
    source_references: tuple[SourceReference, ...] = ()


class ContinentDefinition(ContractModel):
    continent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    reinforcement_bonus: NonNegativeInt
    source_references: tuple[SourceReference, ...] = ()


class Adjacency(ContractModel):
    first_territory_id: str = Field(min_length=1)
    second_territory_id: str = Field(min_length=1)
    source_references: tuple[SourceReference, ...] = ()


class MapSpec(ContractModel):
    map_id: str = Field(min_length=1)
    version: SemanticVersion
    territories: tuple[TerritoryDefinition, ...] = Field(min_length=1)
    continents: tuple[ContinentDefinition, ...] = Field(min_length=1)
    adjacencies: tuple[Adjacency, ...]
    source_references: tuple[SourceReference, ...] = ()

    @model_validator(mode="after")
    def validate_map_integrity(self) -> MapSpec:
        territory_ids = tuple(territory.territory_id for territory in self.territories)
        continent_ids = tuple(continent.continent_id for continent in self.continents)
        _ensure_unique(territory_ids, "Territory IDs")
        _ensure_unique(continent_ids, "Continent IDs")
        _ensure_unique(
            (continent.name for continent in self.continents),
            "Continent names",
            normalizer=_normalize_text,
        )

        names_and_aliases: list[str] = []
        for territory in self.territories:
            names_and_aliases.append(territory.canonical_name)
            names_and_aliases.extend(territory.aliases)
        _ensure_unique(
            names_and_aliases,
            "Territory names and aliases",
            normalizer=_normalize_text,
        )

        known_continents = set(continent_ids)
        unknown_continents = sorted(
            {
                territory.continent_id
                for territory in self.territories
                if territory.continent_id not in known_continents
            }
        )
        if unknown_continents:
            raise ValueError(
                "Territories reference unknown continent IDs: "
                f"{', '.join(unknown_continents)}."
            )

        populated_continents = {
            territory.continent_id for territory in self.territories
        }
        empty_continents = sorted(known_continents - populated_continents)
        if empty_continents:
            raise ValueError(
                f"Continents without territories: {', '.join(empty_continents)}."
            )

        known_territories = set(territory_ids)
        neighbors: dict[str, set[str]] = {
            territory_id: set() for territory_id in territory_ids
        }
        edge_keys: set[tuple[str, str]] = set()
        for adjacency in self.adjacencies:
            first = adjacency.first_territory_id
            second = adjacency.second_territory_id
            if first not in known_territories or second not in known_territories:
                unknown = sorted(
                    {
                        value
                        for value in (first, second)
                        if value not in known_territories
                    }
                )
                raise ValueError(
                    f"Adjacency references unknown territory IDs: {', '.join(unknown)}."
                )
            if first == second:
                raise ValueError(f"Territory {first!r} cannot be adjacent to itself.")
            edge_key = (min(first, second), max(first, second))
            if edge_key in edge_keys:
                raise ValueError(
                    f"Duplicate adjacency between {edge_key[0]} and {edge_key[1]}."
                )
            edge_keys.add(edge_key)
            neighbors[first].add(second)
            neighbors[second].add(first)

        isolated = sorted(
            territory_id for territory_id, adjacent in neighbors.items() if not adjacent
        )
        if isolated:
            raise ValueError(f"Territories without adjacencies: {', '.join(isolated)}.")

        first_territory_id = territory_ids[0]
        reachable = {first_territory_id}
        pending = [first_territory_id]
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current] - reachable:
                reachable.add(neighbor)
                pending.append(neighbor)
        if reachable != known_territories:
            disconnected = sorted(known_territories - reachable)
            raise ValueError(
                f"Map graph is disconnected; unreachable territories: "
                f"{', '.join(disconnected)}."
            )
        return self


class DeckSpec(ContractModel):
    deck_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    card_ids: tuple[str, ...] = Field(min_length=1)
    source_references: tuple[SourceReference, ...] = ()

    @model_validator(mode="after")
    def validate_card_ids(self) -> DeckSpec:
        if any(not card_id.strip() for card_id in self.card_ids):
            raise ValueError("Card IDs must not be blank.")
        _ensure_unique(self.card_ids, "Card IDs")
        return self


class ObjectiveDefinition(ContractModel):
    objective_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    certified_rule_ids: tuple[str, ...] = Field(min_length=1)
    source_references: tuple[SourceReference, ...] = ()

    @model_validator(mode="after")
    def validate_objective_rule_ids(self) -> ObjectiveDefinition:
        _ensure_unique(
            self.certified_rule_ids,
            f"Certified rule IDs for objective {self.objective_id!r}",
        )
        return self


class BoardSpec(ContractModel):
    contract_type: Literal["board_spec"]
    board_id: str = Field(min_length=1)
    version: SemanticVersion
    map: MapSpec
    territory_deck: DeckSpec
    objective_deck: DeckSpec
    objectives: tuple[ObjectiveDefinition, ...] = Field(min_length=1)
    wildcard_card_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_references: tuple[SourceReference, ...] = ()

    @model_validator(mode="after")
    def validate_board_integrity(self) -> BoardSpec:
        _require_source_references(self.source_references, "BoardSpec")
        _require_source_references(self.map.source_references, "MapSpec")
        _require_source_references(
            self.territory_deck.source_references, "Territory deck"
        )
        _require_source_references(
            self.objective_deck.source_references, "Objective deck"
        )
        for territory in self.map.territories:
            _require_source_references(
                territory.source_references,
                f"Territory {territory.territory_id!r}",
            )
            if territory.card_symbol is None:
                raise ValueError(
                    f"Territory {territory.territory_id!r} has no card symbol."
                )
        for continent in self.map.continents:
            _require_source_references(
                continent.source_references,
                f"Continent {continent.continent_id!r}",
            )
        for adjacency in self.map.adjacencies:
            _require_source_references(
                adjacency.source_references,
                "Adjacency "
                f"{adjacency.first_territory_id!r}-"
                f"{adjacency.second_territory_id!r}",
            )
        for objective in self.objectives:
            _require_source_references(
                objective.source_references,
                f"Objective {objective.objective_id!r}",
            )

        _ensure_unique(
            (self.territory_deck.deck_id, self.objective_deck.deck_id), "Deck IDs"
        )
        _validate_objectives(self.objectives)
        _ensure_unique(self.wildcard_card_ids, "Wildcard card IDs")
        if any(not card_id.strip() for card_id in self.wildcard_card_ids):
            raise ValueError("Wildcard card IDs must not be blank.")

        territory_ids = {territory.territory_id for territory in self.map.territories}
        if territory_ids.intersection(self.wildcard_card_ids):
            raise ValueError("Wildcard card IDs must not duplicate territory IDs.")
        expected_territory_card_ids = territory_ids.union(self.wildcard_card_ids)
        if set(self.territory_deck.card_ids) != expected_territory_card_ids:
            missing = sorted(
                expected_territory_card_ids - set(self.territory_deck.card_ids)
            )
            unknown = sorted(
                set(self.territory_deck.card_ids) - expected_territory_card_ids
            )
            raise ValueError(
                "Territory deck must contain exactly one card for every territory "
                "and each wildcard; "
                f"missing={missing}, unknown={unknown}."
            )

        objective_ids = {objective.objective_id for objective in self.objectives}
        if set(self.objective_deck.card_ids) != objective_ids:
            missing = sorted(objective_ids - set(self.objective_deck.card_ids))
            unknown = sorted(set(self.objective_deck.card_ids) - objective_ids)
            raise ValueError(
                "Objective deck must contain exactly one card for every objective; "
                f"missing={missing}, unknown={unknown}."
            )
        return self


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
    source_references: tuple[SourceReference, ...] = ()

    @model_validator(mode="after")
    def validate_ruleset_references(self) -> RulesetSpec:
        _ensure_unique((deck.deck_id for deck in self.decks), "Deck IDs")
        _validate_objectives(self.objectives)
        _ensure_unique(self.certified_rule_ids, "Certified rule IDs")
        known_rule_ids = set(self.certified_rule_ids)
        for objective in self.objectives:
            unknown_rule_ids = sorted(
                set(objective.certified_rule_ids) - known_rule_ids
            )
            if unknown_rule_ids:
                raise ValueError(
                    f"Objective {objective.objective_id!r} references unknown "
                    f"certified rule IDs: {', '.join(unknown_rule_ids)}."
                )
        return self


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


class ReinforcementPool(ContractModel):
    pool_id: str
    player_id: str
    origin: Literal[
        "territory_count",
        "continent_bonus",
        "card_trade",
        "territory_card",
    ]
    remaining_armies: PositiveInt
    allowed_territory_ids: tuple[str, ...] | None = None
    source_card_id: str | None = None

    @model_validator(mode="after")
    def validate_pool_origin(self) -> ReinforcementPool:
        if self.allowed_territory_ids is not None:
            _ensure_unique(
                self.allowed_territory_ids,
                f"Allowed territory IDs for pool {self.pool_id!r}",
            )
            if not self.allowed_territory_ids:
                raise ValueError("Allowed territory IDs cannot be empty.")
        if self.origin == "continent_bonus" and not self.allowed_territory_ids:
            raise ValueError(
                "Continent bonus pools must restrict eligible territories."
            )
        if self.origin == "territory_card":
            if self.source_card_id is None or self.allowed_territory_ids is None:
                raise ValueError(
                    "Territory-card pools require a source card and eligible territory."
                )
            if len(self.allowed_territory_ids) != 1:
                raise ValueError(
                    "Territory-card pools must target exactly one territory."
                )
        return self


class MovementTransfer(ContractModel):
    origin_territory_id: str
    target_territory_id: str
    army_count: PositiveInt


class MovementControl(ContractModel):
    moved_transfers: tuple[MovementTransfer, ...] = ()

    @model_validator(mode="after")
    def validate_movement_ledger(self) -> MovementControl:
        routes = tuple(
            (transfer.origin_territory_id, transfer.target_territory_id)
            for transfer in self.moved_transfers
        )
        if len(routes) != len(set(routes)):
            raise ValueError("Movement transfer routes must be unique.")
        for transfer in self.moved_transfers:
            if transfer.origin_territory_id == transfer.target_territory_id:
                raise ValueError("Movement cannot target its origin territory.")
        return self


class PendingCombat(ContractModel):
    contract_type: Literal["pending_combat"]
    request_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    attacker_player_id: str = Field(min_length=1)
    origin_territory_id: str = Field(min_length=1)
    target_territory_id: str = Field(min_length=1)
    attacker_dice_count: Annotated[int, Field(ge=1, le=3)]
    defender_dice_count: Annotated[int, Field(ge=1, le=3)]

    @model_validator(mode="after")
    def validate_combat_route(self) -> PendingCombat:
        if self.origin_territory_id == self.target_territory_id:
            raise ValueError("Combat cannot target its origin territory.")
        return self


class PendingOccupation(ContractModel):
    contract_type: Literal["pending_occupation"]
    attacker_player_id: str = Field(min_length=1)
    defender_player_id: str = Field(min_length=1)
    origin_territory_id: str = Field(min_length=1)
    target_territory_id: str = Field(min_length=1)
    maximum_army_count: PositiveInt

    @model_validator(mode="after")
    def validate_occupation_route(self) -> PendingOccupation:
        if self.origin_territory_id == self.target_territory_id:
            raise ValueError("Occupation cannot target its origin territory.")
        return self


class PendingObjectiveCheck(ContractModel):
    contract_type: Literal["pending_objective_check"]
    actor_player_id: str = Field(min_length=1)
    territory_id: str = Field(min_length=1)
    resume_phase_id: ObjectiveResumePhaseId
    resume_decision_player_id: str = Field(min_length=1)
    eliminated_player_id: str | None = None


class AppliedChanceOutcome(ContractModel):
    request_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    visibility: Literal["public", "private"]
    outcome_source: Literal["simulation", "site", "physical_board"]


class RandomStreamState(ContractModel):
    stream_id: str = Field(min_length=1)
    bit_generator_state: dict[str, JsonValue]


class RandomStreamsState(ContractModel):
    schema_version: Literal["1.0"]
    algorithm: Literal["PCG64"]
    root_seed: NonNegativeInt
    streams: tuple[RandomStreamState, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_stream_ids(self) -> RandomStreamsState:
        stream_ids = tuple(stream.stream_id for stream in self.streams)
        _ensure_unique(stream_ids, "Random stream IDs")
        required_streams = {"preparation", "objectives", "cards", "combat"}
        missing_streams = sorted(required_streams - set(stream_ids))
        if missing_streams:
            raise ValueError(
                f"Missing required random streams: {', '.join(missing_streams)}."
            )
        invalid_streams = sorted(
            stream_id
            for stream_id in stream_ids
            if stream_id not in required_streams and not stream_id.startswith("policy:")
        )
        if invalid_streams:
            raise ValueError(
                f"Unsupported random stream IDs: {', '.join(invalid_streams)}."
            )
        if any(
            stream_id.startswith("policy:")
            and not stream_id.removeprefix("policy:").strip()
            for stream_id in stream_ids
        ):
            raise ValueError("Policy random streams require a policy ID.")
        return self


class PublicPlayerState(ContractModel):
    player_id: str
    color: str
    territory_count: NonNegativeInt
    army_count: NonNegativeInt
    is_eliminated: bool
    eliminated_by_player_id: str | None = None


class PlayerPrivateState(ContractModel):
    player_id: str
    objective_id: str | None
    objective_fallback_id: str | None = None
    hand_card_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_private_cards(self) -> PlayerPrivateState:
        _ensure_unique(
            self.hand_card_ids, f"Hand card IDs for player {self.player_id!r}"
        )
        return self


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
    hand_limit_discard_card_ids: tuple[str, ...] = ()
    trade_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_card_zones(self) -> DeckState:
        _ensure_unique(
            (
                *self.draw_pile_card_ids,
                *self.discard_pile_card_ids,
                *self.hand_limit_discard_card_ids,
            ),
            "Card IDs across deck zones",
        )
        return self


def _validate_game_state_invariants(
    players: tuple[PublicPlayerState, ...],
    territories: tuple[TerritoryState, ...],
    active_player_id: str | None,
    turn_player_id: str | None,
    decision_player_id: str | None,
    initial_placement_order: tuple[str, ...],
    reinforcement_pools: tuple[ReinforcementPool, ...],
    movement_control: MovementControl,
    is_finished: bool,
) -> None:
    player_ids = tuple(player.player_id for player in players)
    territory_ids = tuple(territory.territory_id for territory in territories)
    _ensure_unique(player_ids, "Player IDs")
    _ensure_unique(
        (player.color for player in players),
        "Player colors",
        normalizer=_normalize_text,
    )
    _ensure_unique(territory_ids, "Territory state IDs")

    known_players = {player.player_id: player for player in players}
    known_territories = set(territory_ids)
    territory_counts = dict.fromkeys(player_ids, 0)
    army_counts = dict.fromkeys(player_ids, 0)
    for territory in territories:
        owner_id = territory.owner_player_id
        if owner_id is None:
            if territory.army_count:
                raise ValueError("Unowned territories cannot contain armies.")
            continue
        owner = known_players.get(owner_id)
        if owner is None:
            raise ValueError(
                f"Territory {territory.territory_id!r} references unknown player "
                f"{owner_id!r}."
            )
        if owner.is_eliminated:
            raise ValueError("Eliminated players cannot own territories.")
        if territory.army_count == 0:
            raise ValueError("Owned territories must contain at least one army.")
        territory_counts[owner_id] += 1
        army_counts[owner_id] += territory.army_count

    for player in players:
        actual_territories = territory_counts[player.player_id]
        actual_armies = army_counts[player.player_id]
        if player.territory_count != actual_territories:
            raise ValueError(
                f"Territory count for player {player.player_id!r} does not match "
                "territory ownership."
            )
        if player.army_count != actual_armies:
            raise ValueError(
                f"Army count for player {player.player_id!r} does not match "
                "territory army counts."
            )
        if player.is_eliminated != (actual_territories == 0):
            raise ValueError(
                f"Elimination status for player {player.player_id!r} does not match "
                "territory ownership."
            )
        eliminator_id = player.eliminated_by_player_id
        if eliminator_id is not None:
            if not player.is_eliminated:
                raise ValueError("Only eliminated players can have an eliminator.")
            if eliminator_id == player.player_id:
                raise ValueError("A player cannot eliminate themselves.")
            if eliminator_id not in known_players:
                raise ValueError("Eliminator must reference a known player.")

    effective_turn_player_id = turn_player_id or active_player_id
    if (
        active_player_id is not None
        and turn_player_id is not None
        and active_player_id != turn_player_id
    ):
        raise ValueError("active_player_id must match turn_player_id when both exist.")
    effective_decision_player_id = decision_player_id or active_player_id
    for label, player_id in (
        ("turn", effective_turn_player_id),
        ("decision", effective_decision_player_id),
    ):
        if player_id is None:
            continue
        resolved_player = known_players.get(player_id)
        if resolved_player is None or resolved_player.is_eliminated:
            raise ValueError(
                f"The {label} player must reference a known, active player."
            )

    _ensure_unique(initial_placement_order, "Initial placement order")
    unknown_order_players = sorted(set(initial_placement_order) - known_players.keys())
    if unknown_order_players:
        raise ValueError(
            "Initial placement order references unknown players: "
            f"{', '.join(unknown_order_players)}."
        )
    if initial_placement_order and len(initial_placement_order) != len(player_ids):
        raise ValueError("Initial placement order must include every player.")

    _ensure_unique(
        (pool.pool_id for pool in reinforcement_pools), "Reinforcement pool IDs"
    )
    for pool in reinforcement_pools:
        resolved_player = known_players.get(pool.player_id)
        if resolved_player is None or resolved_player.is_eliminated:
            raise ValueError(
                f"Reinforcement pool {pool.pool_id!r} references an inactive player."
            )
        if pool.allowed_territory_ids is not None:
            unknown_territories = sorted(
                set(pool.allowed_territory_ids) - known_territories
            )
            if unknown_territories:
                raise ValueError(
                    f"Reinforcement pool {pool.pool_id!r} references unknown "
                    f"territories: {', '.join(unknown_territories)}."
                )

    moved_in_counts = dict.fromkeys(known_territories, 0)
    for transfer in movement_control.moved_transfers:
        if (
            transfer.origin_territory_id not in known_territories
            or transfer.target_territory_id not in known_territories
        ):
            raise ValueError("Movement control references an unknown territory.")
        moved_in_counts[transfer.target_territory_id] += transfer.army_count
    for territory in territories:
        if moved_in_counts[territory.territory_id] > territory.army_count:
            raise ValueError(
                f"Movement ledger exceeds armies in territory "
                f"{territory.territory_id!r}."
            )
    if not is_finished and not players and active_player_id is not None:
        raise ValueError("An active game must contain players.")


def _validate_game_phase(
    *,
    phase_id: GamePhaseId,
    turn_number: int,
    is_finished: bool,
    active_player_id: str | None,
    turn_player_id: str | None,
    decision_player_id: str | None,
    has_conquered_territory_this_turn: bool,
) -> None:
    if is_finished != (phase_id == "finished"):
        raise ValueError("The finished phase and is_finished flag must agree.")
    if is_finished and any(
        player_id is not None
        for player_id in (
            active_player_id,
            turn_player_id,
            decision_player_id,
        )
    ):
        raise ValueError("Finished games cannot have an active player decision.")
    if phase_id in {
        "reinforcement",
        "reinforcement_placement",
        "attack",
        "occupation",
        "movement",
        "card_purchase",
        "card_discard",
        "turn_end",
    } and turn_number < 2:
        raise ValueError("Regular-turn phases cannot occur before round two.")
    if phase_id in {"card_purchase", "card_discard"} and (
        not has_conquered_territory_this_turn
    ):
        raise ValueError(
            "Card purchase and discard phases require a territory conquest this turn."
        )


def _validate_pending_decisions(
    *,
    players: tuple[PublicPlayerState, ...],
    phase_id: GamePhaseId,
    decision_player_id: str | None,
    territories: tuple[TerritoryState, ...],
    pending_combat: PendingCombat | None,
    pending_occupation: PendingOccupation | None,
    pending_objective_check: PendingObjectiveCheck | None,
) -> None:
    pending_decisions = sum(
        item is not None
        for item in (pending_combat, pending_occupation, pending_objective_check)
    )
    objective_check_before_occupation = (
        pending_occupation is not None
        and pending_objective_check is not None
        and pending_objective_check.resume_phase_id == "occupation"
    )
    if pending_decisions > 1 and not objective_check_before_occupation:
        raise ValueError("Only one game decision can be pending at a time.")
    territory_by_id = {territory.territory_id: territory for territory in territories}
    if pending_combat is not None:
        if phase_id != "attack":
            raise ValueError("Pending combat is only valid during the attack phase.")
        if decision_player_id != pending_combat.attacker_player_id:
            raise ValueError("Pending combat must belong to the decision player.")
        origin = territory_by_id.get(pending_combat.origin_territory_id)
        target = territory_by_id.get(pending_combat.target_territory_id)
        if origin is None or target is None:
            raise ValueError("Pending combat references an unknown territory.")
        if origin.owner_player_id != pending_combat.attacker_player_id:
            raise ValueError("Pending combat origin must belong to the attacker.")
        if (
            target.owner_player_id is None
            or target.owner_player_id == pending_combat.attacker_player_id
        ):
            raise ValueError("Pending combat target must belong to an opponent.")
        if pending_combat.attacker_dice_count > min(3, origin.army_count - 1):
            raise ValueError("Pending combat exceeds the attacking armies.")
        if pending_combat.defender_dice_count != min(target.army_count, 3):
            raise ValueError("Pending combat has an invalid defender dice count.")
    if pending_occupation is not None:
        objective_check_before_occupation = (
            phase_id == "objective_check"
            and pending_objective_check is not None
            and pending_objective_check.resume_phase_id == "occupation"
        )
        if phase_id != "occupation" and not objective_check_before_occupation:
            raise ValueError(
                "Pending occupation is only valid during the occupation phase."
            )
        if decision_player_id != pending_occupation.attacker_player_id:
            raise ValueError("Pending occupation must belong to the decision player.")
        known_player_ids = {player.player_id for player in players}
        if (
            pending_occupation.defender_player_id == pending_occupation.attacker_player_id
            or pending_occupation.defender_player_id not in known_player_ids
        ):
            raise ValueError("Pending occupation must reference a different defender.")
        defender = next(
            player
            for player in players
            if player.player_id == pending_occupation.defender_player_id
        )
        if (
            defender.is_eliminated
            and defender.eliminated_by_player_id
            != pending_occupation.attacker_player_id
        ):
            raise ValueError(
                "Eliminated defender must be attributed to the pending attacker."
            )
        origin = territory_by_id.get(pending_occupation.origin_territory_id)
        target = territory_by_id.get(pending_occupation.target_territory_id)
        if origin is None or target is None:
            raise ValueError("Pending occupation references an unknown territory.")
        if origin.owner_player_id != pending_occupation.attacker_player_id:
            raise ValueError("Pending occupation origin must belong to the attacker.")
        if target.owner_player_id is not None or target.army_count != 0:
            raise ValueError("Pending occupation target must be unowned and empty.")
        if pending_occupation.maximum_army_count > origin.army_count - 1:
            raise ValueError("Pending occupation exceeds the available armies.")
    elif phase_id == "occupation":
        raise ValueError("Occupation phase requires a pending occupation.")
    if pending_objective_check is not None:
        if phase_id != "objective_check":
            raise ValueError(
                "A pending objective check is only valid in its dedicated phase."
            )
        if decision_player_id != pending_objective_check.resume_decision_player_id:
            raise ValueError(
                "Pending objective check has an inconsistent resumption player."
            )
        known_player_ids = {player.player_id for player in players}
        if pending_objective_check.actor_player_id not in known_player_ids:
            raise ValueError("Pending objective check references an unknown actor.")
        target = territory_by_id.get(pending_objective_check.territory_id)
        if target is None:
            raise ValueError("Pending objective check references an unknown territory.")
        if pending_objective_check.resume_phase_id == "occupation":
            if (
                pending_occupation is None
                or pending_occupation.target_territory_id
                != pending_objective_check.territory_id
            ):
                raise ValueError(
                    "Objective check before occupation requires its pending conquest."
                )
        elif pending_occupation is not None:
            raise ValueError(
                "Pending occupation must resume only after objective checking."
            )
        if pending_objective_check.eliminated_player_id is not None:
            eliminated_player = next(
                (
                    player
                    for player in players
                    if player.player_id == pending_objective_check.eliminated_player_id
                ),
                None,
            )
            if (
                eliminated_player is None
                or not eliminated_player.is_eliminated
                or eliminated_player.eliminated_by_player_id
                != pending_objective_check.actor_player_id
            ):
                raise ValueError(
                    "Pending objective check has an inconsistent elimination."
                )
    elif phase_id == "objective_check":
        raise ValueError("Objective-check phase requires a pending objective check.")


class FullGameState(ContractModel):
    contract_type: Literal["full_game_state"]
    game_id: str
    ruleset_id: str
    rules_version: SemanticVersion
    state_version: NonNegativeInt
    phase_id: GamePhaseId
    turn_number: NonNegativeInt
    has_conquered_territory_this_turn: bool = False
    active_player_id: str | None
    turn_player_id: str | None = None
    decision_player_id: str | None = None
    initial_placement_order: tuple[str, ...] = ()
    players: tuple[FullPlayerState, ...]
    territories: tuple[TerritoryState, ...]
    reinforcement_pools: tuple[ReinforcementPool, ...] = ()
    movement_control: MovementControl = Field(default_factory=MovementControl)
    pending_combat: PendingCombat | None = None
    pending_occupation: PendingOccupation | None = None
    pending_objective_check: PendingObjectiveCheck | None = None
    applied_chance_outcomes: tuple[AppliedChanceOutcome, ...] = ()
    winner_player_id: str | None = None
    deck: DeckState
    engine_random_state: JsonValue
    random_streams_state: RandomStreamsState | None = None
    is_finished: bool

    @model_validator(mode="after")
    def validate_full_state(self) -> FullGameState:
        if (
            self.random_streams_state is not None
            and self.engine_random_state is not None
        ):
            raise ValueError(
                "Use random_streams_state instead of the legacy engine_random_state."
            )
        public_players = tuple(player.public for player in self.players)
        _validate_game_state_invariants(
            players=public_players,
            territories=self.territories,
            active_player_id=self.active_player_id,
            turn_player_id=self.turn_player_id,
            decision_player_id=self.decision_player_id,
            initial_placement_order=self.initial_placement_order,
            reinforcement_pools=self.reinforcement_pools,
            movement_control=self.movement_control,
            is_finished=self.is_finished,
        )
        _validate_game_phase(
            phase_id=self.phase_id,
            turn_number=self.turn_number,
            is_finished=self.is_finished,
            active_player_id=self.active_player_id,
            turn_player_id=self.turn_player_id,
            decision_player_id=self.decision_player_id,
            has_conquered_territory_this_turn=(
                self.has_conquered_territory_this_turn
            ),
        )
        _validate_pending_decisions(
            players=public_players,
            phase_id=self.phase_id,
            decision_player_id=self.decision_player_id or self.active_player_id,
            territories=self.territories,
            pending_combat=self.pending_combat,
            pending_occupation=self.pending_occupation,
            pending_objective_check=self.pending_objective_check,
        )
        if self.winner_player_id is not None:
            winner = next(
                (
                    player
                    for player in self.players
                    if player.public.player_id == self.winner_player_id
                ),
                None,
            )
            if (
                not self.is_finished
                or winner is None
                or winner.public.is_eliminated
            ):
                raise ValueError("Winner must be an active player in a finished game.")
            if winner.private.objective_id is None:
                raise ValueError("A winner must have an assigned objective to reveal.")
        elif self.is_finished:
            raise ValueError("Finished games require an official winner.")
        objectives = tuple(
            player.private.objective_id
            for player in self.players
            if player.private.objective_id is not None
        )
        _ensure_unique(objectives, "Assigned objective IDs")
        decision_player_id = self.decision_player_id or self.active_player_id
        if self.phase_id == "card_discard" and decision_player_id is None:
            raise ValueError("Card-discard phase requires a decision player.")
        for player in self.players:
            hand_size = len(player.private.hand_card_ids)
            if (
                self.phase_id == "card_discard"
                and player.public.player_id == decision_player_id
            ):
                if hand_size != 6:
                    raise ValueError(
                        "Card-discard phase requires exactly six cards in the "
                        "decision player's hand."
                    )
            elif hand_size > 5:
                raise ValueError(
                    "Player hands cannot exceed five cards outside card-discard."
                )
        cards = [
            *self.deck.draw_pile_card_ids,
            *self.deck.discard_pile_card_ids,
            *self.deck.hand_limit_discard_card_ids,
            *(
                card_id
                for player in self.players
                for card_id in player.private.hand_card_ids
            ),
        ]
        _ensure_unique(cards, "Card IDs across game zones and player hands")
        _ensure_unique(
            (outcome.request_id for outcome in self.applied_chance_outcomes),
            "Applied chance request IDs",
        )
        _ensure_unique(
            (outcome.event_id for outcome in self.applied_chance_outcomes),
            "Applied chance event IDs",
        )
        return self


class PublicGameState(ContractModel):
    contract_type: Literal["public_game_state"]
    game_id: str
    ruleset_id: str
    rules_version: SemanticVersion
    state_version: NonNegativeInt
    phase_id: GamePhaseId
    turn_number: NonNegativeInt
    has_conquered_territory_this_turn: bool = False
    active_player_id: str | None
    turn_player_id: str | None = None
    decision_player_id: str | None = None
    initial_placement_order: tuple[str, ...] = ()
    players: tuple[PublicPlayerState, ...]
    territories: tuple[TerritoryState, ...]
    reinforcement_pools: tuple[ReinforcementPool, ...] = ()
    movement_control: MovementControl = Field(default_factory=MovementControl)
    pending_combat: PendingCombat | None = None
    pending_occupation: PendingOccupation | None = None
    pending_objective_check: PendingObjectiveCheck | None = None
    applied_chance_outcomes: tuple[AppliedChanceOutcome, ...] = ()
    winner_player_id: str | None = None
    winner_objective_id: str | None = None
    winner_objective_fallback_id: str | None = None
    is_finished: bool

    @model_validator(mode="after")
    def validate_public_state(self) -> PublicGameState:
        _validate_game_state_invariants(
            players=self.players,
            territories=self.territories,
            active_player_id=self.active_player_id,
            turn_player_id=self.turn_player_id,
            decision_player_id=self.decision_player_id,
            initial_placement_order=self.initial_placement_order,
            reinforcement_pools=self.reinforcement_pools,
            movement_control=self.movement_control,
            is_finished=self.is_finished,
        )
        _validate_game_phase(
            phase_id=self.phase_id,
            turn_number=self.turn_number,
            is_finished=self.is_finished,
            active_player_id=self.active_player_id,
            turn_player_id=self.turn_player_id,
            decision_player_id=self.decision_player_id,
            has_conquered_territory_this_turn=(
                self.has_conquered_territory_this_turn
            ),
        )
        _validate_pending_decisions(
            players=self.players,
            phase_id=self.phase_id,
            decision_player_id=self.decision_player_id or self.active_player_id,
            territories=self.territories,
            pending_combat=self.pending_combat,
            pending_occupation=self.pending_occupation,
            pending_objective_check=self.pending_objective_check,
        )
        if any(
            outcome.visibility != "public"
            for outcome in self.applied_chance_outcomes
        ):
            raise ValueError("Public state cannot contain private chance outcome IDs.")
        _ensure_unique(
            (outcome.request_id for outcome in self.applied_chance_outcomes),
            "Applied public chance request IDs",
        )
        _ensure_unique(
            (outcome.event_id for outcome in self.applied_chance_outcomes),
            "Applied public chance event IDs",
        )
        if self.winner_player_id is not None:
            winner = next(
                (
                    player
                    for player in self.players
                    if player.player_id == self.winner_player_id
                ),
                None,
            )
            if not self.is_finished or winner is None or winner.is_eliminated:
                raise ValueError("Winner must be an active player in a finished game.")
            if self.winner_objective_id is None:
                raise ValueError("Finished public states must reveal the winner's objective.")
        elif (
            self.winner_objective_id is not None
            or self.winner_objective_fallback_id is not None
        ):
            raise ValueError("Winner objectives are visible only after a victory.")
        elif self.is_finished:
            raise ValueError("Finished public states require an official winner.")
        return self


class ArmyPlacement(ContractModel):
    territory_id: str
    army_count: Literal[1]
    source_pool_id: str | None = None


class ClaimTerritoryOperation(ContractModel):
    action_type: Literal["claim_territory"]
    territory_id: str


class PlaceArmiesOperation(ContractModel):
    action_type: Literal["place_armies"]
    placements: tuple[ArmyPlacement, ...] = Field(min_length=1, max_length=1)
    placement_phase: Literal["initial", "reinforcement"]


class TradeCardsOperation(ContractModel):
    action_type: Literal["trade_cards"]
    card_ids: tuple[str, str, str]


class DrawTerritoryCardOperation(ContractModel):
    action_type: Literal["draw_territory_card"]


class DiscardTerritoryCardOperation(ContractModel):
    action_type: Literal["discard_territory_card"]
    card_id: str = Field(min_length=1)


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


class EndPhaseOperation(ContractModel):
    action_type: Literal["end_phase"]
    phase_id: EndablePhaseId


class EndTurnOperation(ContractModel):
    action_type: Literal["end_turn"]


ActionOperation = Annotated[
    ClaimTerritoryOperation
    | PlaceArmiesOperation
    | TradeCardsOperation
    | DrawTerritoryCardOperation
    | DiscardTerritoryCardOperation
    | AttackOperation
    | OccupyTerritoryOperation
    | MoveArmiesOperation
    | EndPhaseOperation
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
            raise ValueError(
                "Player observation may contain only its own private state."
            )
        decision_player_id = (
            self.public_state.decision_player_id or self.public_state.active_player_id
        )
        if (
            self.legal_actions
            and decision_player_id is not None
            and decision_player_id != self.player_id
        ):
            raise ValueError(
                "Only the player responsible for the next decision may receive "
                "legal actions."
            )
        for action in self.legal_actions:
            if action.actor_player_id != self.player_id:
                raise ValueError("Legal actions must belong to the observed player.")
            if action.expected_state_version != self.state_version:
                raise ValueError(
                    "Legal actions must target the observed state version."
                )
        for event in self.recent_events:
            if event.visibility == "engine":
                raise ValueError(
                    "Engine-private events cannot enter a player observation."
                )
            if (
                event.visibility == "private"
                and self.player_id not in event.visible_to_player_ids
            ):
                raise ValueError(
                    "Observation contains an event private to another player."
                )
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
    outcome_source: Literal["simulation", "site", "physical_board"]
    visible_to_player_ids: tuple[str, ...] = ()


SnapshotState = Annotated[
    PublicGameState | FullGameState | PlayerObservation,
    Field(discriminator="contract_type"),
]


class Snapshot(ContractModel):
    contract_type: Literal["snapshot"]
    snapshot_id: str
    replay_version: SemanticVersion = REPLAY_VERSION
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
        if self.state.game_id != self.game_id:
            raise ValueError("Snapshot and state game IDs must match.")
        if self.state.state_version != self.state_version:
            raise ValueError("Snapshot and state versions must match.")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("Snapshot timestamps must include a timezone.")
        if canonical_sha256(self.state) != self.state_hash:
            raise ValueError("Snapshot state hash does not match canonical state.")
        if (
            self.visibility == "player_private"
            and isinstance(self.state, PlayerObservation)
            and self.subject_player_id != self.state.player_id
        ):
            raise ValueError("Private snapshot subject must match its player.")
        if self.visibility != "player_private" and self.subject_player_id is not None:
            raise ValueError("Only private snapshots may have a subject player.")
        return self


class ReplayActionEntry(ContractModel):
    entry_type: Literal["action"]
    sequence_number: PositiveInt
    action: Action


class ReplayChanceEntry(ContractModel):
    entry_type: Literal["chance_outcome"]
    sequence_number: PositiveInt
    outcome: ChanceOutcome


class OfficialVictoryOutcome(ContractModel):
    outcome_type: Literal["official_victory"]
    winner_player_id: str = Field(min_length=1)


class IndividualEliminationOutcome(ContractModel):
    outcome_type: Literal["individual_elimination"]
    eliminated_player_id: str = Field(min_length=1)


class ResignationOutcome(ContractModel):
    outcome_type: Literal["resignation"]
    player_id: str = Field(min_length=1)


class OperationalInterruptionOutcome(ContractModel):
    outcome_type: Literal["operational_interruption"]
    cause_code: str = Field(min_length=1)


class TrainingTruncationOutcome(ContractModel):
    outcome_type: Literal["training_truncation"]
    limit_type: Literal["turn_count", "action_count", "time_budget_ms"]
    limit_value: PositiveInt


ReplayOutcome = Annotated[
    OfficialVictoryOutcome
    | IndividualEliminationOutcome
    | ResignationOutcome
    | OperationalInterruptionOutcome
    | TrainingTruncationOutcome,
    Field(discriminator="outcome_type"),
]


ReplayEntry = Annotated[
    ReplayActionEntry | ReplayChanceEntry,
    Field(discriminator="entry_type"),
]


def _validate_replay_entries(entries: tuple[ReplayEntry, ...]) -> None:
    sequence_numbers = tuple(entry.sequence_number for entry in entries)
    expected_sequence_numbers = tuple(range(1, len(entries) + 1))
    if sequence_numbers != expected_sequence_numbers:
        raise ValueError("Replay entries must have contiguous sequence numbers from 1.")
    chance_outcomes = tuple(
        entry.outcome
        for entry in entries
        if isinstance(entry, ReplayChanceEntry)
    )
    _ensure_unique(
        (outcome.request_id for outcome in chance_outcomes),
        "Chance request IDs in replay",
    )
    _ensure_unique(
        (outcome.event_id for outcome in chance_outcomes),
        "Chance event IDs in replay",
    )


class FullReplay(ContractModel):
    contract_type: Literal["full_replay"]
    replay_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)
    replay_version: SemanticVersion
    api_version: ApiVersionLiteral
    contract_schema_version: ContractSchemaVersionLiteral
    ruleset_id: str
    rules_version: SemanticVersion
    initial_snapshot: Snapshot
    entries: tuple[ReplayEntry, ...] = ()
    outcome: ReplayOutcome | None = None
    final_state_hash: Sha256Hash

    @model_validator(mode="after")
    def validate_full_replay(self) -> FullReplay:
        if self.initial_snapshot.visibility != "full":
            raise ValueError("Full replays require a full initial snapshot.")
        if not isinstance(self.initial_snapshot.state, FullGameState):
            raise ValueError("Full replay snapshots must contain FullGameState.")
        if self.initial_snapshot.game_id != self.game_id:
            raise ValueError("Replay and initial snapshot game IDs must match.")
        if (
            self.initial_snapshot.state.ruleset_id != self.ruleset_id
            or self.initial_snapshot.state.rules_version != self.rules_version
        ):
            raise ValueError("Replay and initial snapshot rulesets must match.")
        if self.initial_snapshot.replay_version != self.replay_version:
            raise ValueError("Replay and initial snapshot versions must match.")
        _validate_replay_entries(self.entries)
        return self


class PublicReplay(ContractModel):
    contract_type: Literal["public_replay"]
    replay_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)
    replay_version: SemanticVersion
    api_version: ApiVersionLiteral
    contract_schema_version: ContractSchemaVersionLiteral
    ruleset_id: str
    rules_version: SemanticVersion
    initial_snapshot: Snapshot
    entries: tuple[ReplayEntry, ...] = ()
    outcome: ReplayOutcome | None = None
    final_state_hash: Sha256Hash

    @model_validator(mode="after")
    def validate_public_replay(self) -> PublicReplay:
        if self.initial_snapshot.visibility != "public":
            raise ValueError("Public replays require a public initial snapshot.")
        if not isinstance(self.initial_snapshot.state, PublicGameState):
            raise ValueError("Public replay snapshots must contain PublicGameState.")
        if self.initial_snapshot.game_id != self.game_id:
            raise ValueError("Replay and initial snapshot game IDs must match.")
        if (
            self.initial_snapshot.state.ruleset_id != self.ruleset_id
            or self.initial_snapshot.state.rules_version != self.rules_version
        ):
            raise ValueError("Replay and initial snapshot rulesets must match.")
        if self.initial_snapshot.replay_version != self.replay_version:
            raise ValueError("Replay and initial snapshot versions must match.")
        _validate_replay_entries(self.entries)
        private_chance_entries = tuple(
            entry
            for entry in self.entries
            if isinstance(entry, ReplayChanceEntry)
            and entry.outcome.visibility != "public"
        )
        if private_chance_entries:
            raise ValueError("Public replays cannot contain private chance outcomes.")
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
    BoardSpec
    | RulesetSpec
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
    | FullReplay
    | PublicReplay
    | FullReplay
    | PublicReplay
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
    elif contract_type in ("full_replay", "public_replay"):
        initial_snapshot = payload.get("initial_snapshot")
        if isinstance(initial_snapshot, dict):
            _validate_live_payload_versions(initial_snapshot.get("state"))
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("entry_type") == "action":
                    _validate_action_schema_version(entry.get("action"))


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
        raise IncompatibleContractVersionError("API", actual_api_version, API_VERSION)
    _validate_live_payload_versions(decoded.get("payload"))
    return ContractEnvelope.model_validate_json(serialized)


def contract_json_schema() -> dict[str, object]:
    schema: dict[str, object] = ContractEnvelope.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"urn:war:contracts:{CONTRACT_SCHEMA_VERSION}"
    return schema
