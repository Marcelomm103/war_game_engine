"""Protocol-driven human-versus-agent runner for WAR 1 classic."""

from __future__ import annotations

import argparse
import importlib
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

from numpy.random import Generator
from pydantic import Field, JsonValue, ValidationError, model_validator

from .board_spec import WAR1_BOARD_SPEC
from .contracts import (
    Action,
    AgentProtocol,
    AttackOperation,
    BoardSpec,
    ChanceOutcome,
    ClaimTerritoryOperation,
    ContractModel,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    EndPhaseOperation,
    EndTurnOperation,
    FullGameState,
    MoveArmiesOperation,
    OccupyTerritoryOperation,
    PlaceArmiesOperation,
    PlayerObservation,
    Snapshot,
    TradeCardsOperation,
)
from .engine import (
    apply_action,
    apply_combat_outcome,
    legal_actions,
    simulate_pending_combat_outcome,
)
from .objectives import PlayerColor
from .randomness import InvalidRandomStreamStateError, RandomStreams
from .replay import SnapshotRestoreError, create_snapshot, restore_snapshot
from .setup import PLAYER_COUNT_RANGE, InvalidSetupError, SetupPlayer, prepare_game
from .state import observe_player, to_public_game_state
from .versioning import (
    ACTION_ENCODING_VERSION,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_ENCODING_VERSION,
    OBSERVATION_SCHEMA_VERSION,
)

AGENT_PROTOCOL_VERSION = "1.0.0"
RUNNER_SAVE_SCHEMA_VERSION: Literal["1.0"] = "1.0"
PLAYER_COLORS: tuple[PlayerColor, ...] = (
    "white",
    "red",
    "black",
    "blue",
    "yellow",
    "green",
)


class RunnerError(Exception):
    """Base class for expected runner and configuration errors."""


class RunnerConfigurationError(RunnerError):
    """Raised when a game or model assignment is invalid."""


class AgentLoadError(RunnerError):
    """Raised when an agent factory cannot be loaded."""


class AgentCompatibilityError(RunnerError):
    """Raised when an agent does not support the current runtime contracts."""


class AgentDecisionError(RunnerError):
    """Raised when an agent returns an invalid decision."""


class AgentDecisionBudgetExceeded(AgentDecisionError):
    """Raised when an agent's decision exceeds its declared time budget."""


class RunnerSaveError(RunnerError):
    """Raised when a runner save cannot be parsed or resumed."""


class AgentDecision(ContractModel):
    action: Action
    memory: JsonValue | None = None


@runtime_checkable
class AgentImplementation(Protocol):
    """Runtime adapter implemented by an agent model or a model plugin."""

    @property
    def protocol(self) -> AgentProtocol: ...

    def decide(
        self,
        observation: PlayerObservation,
        *,
        rng: Generator,
        memory: JsonValue | None,
    ) -> AgentDecision: ...


class RunnerSave(ContractModel):
    contract_type: Literal["war_play_save"]
    schema_version: Literal["1.0"]
    snapshot: Snapshot
    human_player_id: str
    model_specs_by_player: dict[str, str]
    agent_protocols_by_model: dict[str, AgentProtocol]
    agent_memory_by_player: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_runner_save(self) -> RunnerSave:
        if self.snapshot.visibility != "full" or not isinstance(
            self.snapshot.state, FullGameState
        ):
            raise ValueError("Runner saves require a full game snapshot.")

        player_ids = {
            player.public.player_id for player in self.snapshot.state.players
        }
        if self.human_player_id not in player_ids:
            raise ValueError("Runner save references an unknown human player.")
        ai_player_ids = player_ids - {self.human_player_id}
        if set(self.model_specs_by_player) != ai_player_ids:
            raise ValueError("Runner save must assign one model to every AI seat.")
        if any(not spec.strip() for spec in self.model_specs_by_player.values()):
            raise ValueError("Runner save contains a blank model specification.")
        model_specs = set(self.model_specs_by_player.values())
        if set(self.agent_protocols_by_model) != model_specs:
            raise ValueError("Runner save agent protocols do not match its models.")
        if not set(self.agent_memory_by_player).issubset(ai_player_ids):
            raise ValueError("Runner save contains memory for a non-AI seat.")
        return self


def _protocol_from_agent(agent: AgentImplementation) -> AgentProtocol:
    protocol: object = agent.protocol
    if not isinstance(protocol, AgentProtocol):
        raise AgentCompatibilityError(
            "Agent implementations must expose an AgentProtocol contract."
        )
    return protocol


def _validate_agent_protocol(
    protocol: AgentProtocol,
    *,
    model_spec: str,
    board_id: str,
) -> None:
    if protocol.protocol_version != AGENT_PROTOCOL_VERSION:
        raise AgentCompatibilityError(
            f"Model {model_spec!r} uses agent protocol "
            f"{protocol.protocol_version!r}; expected {AGENT_PROTOCOL_VERSION!r}."
        )
    if board_id not in protocol.supported_ruleset_ids:
        raise AgentCompatibilityError(
            f"Model {model_spec!r} does not support ruleset {board_id!r}."
        )
    expected_versions = {
        "observation schema": (
            protocol.observation_schema_version,
            OBSERVATION_SCHEMA_VERSION,
        ),
        "observation encoding": (
            protocol.observation_encoding_version,
            OBSERVATION_ENCODING_VERSION,
        ),
        "action schema": (protocol.action_schema_version, ACTION_SCHEMA_VERSION),
        "action encoding": (
            protocol.action_encoding_version,
            ACTION_ENCODING_VERSION,
        ),
    }
    for label, (actual, expected) in expected_versions.items():
        if actual != expected:
            raise AgentCompatibilityError(
                f"Model {model_spec!r} uses {label} {actual!r}; "
                f"expected {expected!r}."
            )
    if (
        protocol.memory_scope != "per_player"
        or protocol.rng_scope != "private_per_player"
    ):
        raise AgentCompatibilityError(
            f"Model {model_spec!r} must use per-player memory and RNG."
        )
    if not protocol.agent_id.strip():
        raise AgentCompatibilityError(
            f"Model {model_spec!r} has a blank agent ID."
        )


def _with_random_streams(
    state: FullGameState,
    random_streams: RandomStreams,
) -> FullGameState:
    state_data = state.model_dump(mode="python")
    state_data["random_streams_state"] = random_streams.snapshot()
    return FullGameState.model_validate(state_data)


class GameSession:
    """Owns the full game state while exposing only player-scoped observations."""

    def __init__(
        self,
        *,
        state: FullGameState,
        human_player_id: str,
        model_specs_by_player: Mapping[str, str],
        models_by_spec: Mapping[str, AgentImplementation],
        agent_memory_by_player: Mapping[str, JsonValue] | None = None,
        expected_protocols_by_model: Mapping[str, AgentProtocol] | None = None,
        board: BoardSpec = WAR1_BOARD_SPEC,
    ) -> None:
        if state.random_streams_state is None:
            raise RunnerConfigurationError(
                "Human-versus-agent sessions require restorable random streams."
            )
        if state.ruleset_id != board.board_id:
            raise RunnerConfigurationError(
                f"Game state uses ruleset {state.ruleset_id!r}, not "
                f"{board.board_id!r}."
            )

        player_ids = {player.public.player_id for player in state.players}
        if human_player_id not in player_ids:
            raise RunnerConfigurationError(
                f"Unknown human player ID {human_player_id!r}."
            )
        ai_player_ids = player_ids - {human_player_id}
        if set(model_specs_by_player) != ai_player_ids:
            raise RunnerConfigurationError(
                "Assign exactly one model specification to every non-human seat."
            )
        required_models = set(model_specs_by_player.values())
        if set(models_by_spec) != required_models:
            raise RunnerConfigurationError(
                "Loaded agent models do not match the seat assignments."
            )
        if expected_protocols_by_model is not None and set(
            expected_protocols_by_model
        ) != required_models:
            raise RunnerSaveError(
                "Saved agent protocols do not match the assigned models."
            )

        protocols: dict[str, AgentProtocol] = {}
        for model_spec, agent in models_by_spec.items():
            if not isinstance(agent, AgentImplementation):
                raise AgentCompatibilityError(
                    f"Model {model_spec!r} must implement AgentImplementation."
                )
            protocol = _protocol_from_agent(agent)
            _validate_agent_protocol(
                protocol,
                model_spec=model_spec,
                board_id=board.board_id,
            )
            if (
                expected_protocols_by_model is not None
                and protocol != expected_protocols_by_model[model_spec]
            ):
                raise AgentCompatibilityError(
                    f"Model {model_spec!r} no longer matches the saved "
                    "AgentProtocol."
                )
            protocols[model_spec] = protocol

        memories = dict(agent_memory_by_player or {})
        if not set(memories).issubset(ai_player_ids):
            raise RunnerConfigurationError(
                "Agent memory can only be assigned to non-human seats."
            )

        self.state = state
        self.human_player_id = human_player_id
        self.model_specs_by_player = dict(model_specs_by_player)
        self.models_by_spec = dict(models_by_spec)
        self.agent_protocols_by_model = protocols
        self.agent_memory_by_player = memories
        self.board = board

    @property
    def decision_player_id(self) -> str | None:
        return self.state.decision_player_id or self.state.active_player_id

    def available_actions(self) -> tuple[Action, ...]:
        return legal_actions(self.state, board=self.board)

    def human_observation(self) -> PlayerObservation:
        if self.decision_player_id != self.human_player_id:
            raise RunnerConfigurationError(
                "The human is not responsible for the current decision."
            )
        return observe_player(
            self.state,
            self.human_player_id,
            legal_actions=self.available_actions(),
        )

    def apply_human_action(self, action: Action) -> None:
        if self.decision_player_id != self.human_player_id:
            raise RunnerConfigurationError(
                "The human is not responsible for the current decision."
            )
        if action not in self.available_actions():
            raise AgentDecisionError(
                "Selected action is not in the current legal-action list."
            )
        self.state = apply_action(self.state, action, board=self.board)

    def play_agent_turn(self) -> tuple[Action, float]:
        if self.state.is_finished:
            raise RunnerConfigurationError("A finished game has no AI decisions.")
        player_id = self.decision_player_id
        if player_id is None:
            raise RunnerConfigurationError(
                "The current game state has no decision player."
            )
        if player_id == self.human_player_id:
            raise RunnerConfigurationError(
                "The human is responsible for the current decision."
            )
        model_spec = self.model_specs_by_player.get(player_id)
        if model_spec is None:
            raise RunnerConfigurationError(
                f"No AI model is assigned to player {player_id!r}."
            )

        actions = self.available_actions()
        if not actions:
            raise RunnerConfigurationError(
                f"No legal actions are available in phase {self.state.phase_id!r}."
            )
        random_state = self.state.random_streams_state
        if random_state is None:
            raise RunnerConfigurationError(
                "The current game state has no restorable random streams."
            )

        agent = self.models_by_spec[model_spec]
        protocol = self.agent_protocols_by_model[model_spec]
        observation = observe_player(
            self.state,
            player_id,
            legal_actions=actions,
        )
        streams = RandomStreams.restore(random_state)
        policy_rng = streams.policy_generator(f"{protocol.agent_id}:{player_id}")
        started_at = time.perf_counter_ns()
        decision = agent.decide(
            observation,
            rng=policy_rng,
            memory=self.agent_memory_by_player.get(player_id),
        )
        elapsed_ms = (time.perf_counter_ns() - started_at) / 1_000_000
        if elapsed_ms > protocol.decision_budget_ms:
            raise AgentDecisionBudgetExceeded(
                f"Model {protocol.agent_id!r} took {elapsed_ms:.1f} ms; "
                f"its budget is {protocol.decision_budget_ms} ms."
            )
        if not isinstance(decision, AgentDecision):
            raise AgentDecisionError(
                f"Model {protocol.agent_id!r} must return AgentDecision."
            )
        if decision.action not in actions:
            raise AgentDecisionError(
                f"Model {protocol.agent_id!r} returned an action outside its "
                "legal-action observation."
            )

        state_with_policy_rng = _with_random_streams(self.state, streams)
        self.state = apply_action(
            state_with_policy_rng,
            decision.action,
            board=self.board,
        )
        if decision.memory is None:
            self.agent_memory_by_player.pop(player_id, None)
        else:
            self.agent_memory_by_player[player_id] = decision.memory
        return decision.action, elapsed_ms

    def resolve_pending_combat(self) -> ChanceOutcome:
        if self.state.pending_combat is None:
            raise RunnerConfigurationError("There is no pending combat to resolve.")
        outcome, random_streams_state = simulate_pending_combat_outcome(self.state)
        self.state = apply_combat_outcome(
            self.state,
            outcome,
            random_streams_state=random_streams_state,
            board=self.board,
        )
        return outcome

    def create_save(self) -> RunnerSave:
        snapshot = create_snapshot(
            f"{self.state.game_id}:runner-save:{self.state.state_version}",
            self.state,
            visibility="full",
        )
        return RunnerSave(
            contract_type="war_play_save",
            schema_version=RUNNER_SAVE_SCHEMA_VERSION,
            snapshot=snapshot,
            human_player_id=self.human_player_id,
            model_specs_by_player=self.model_specs_by_player,
            agent_protocols_by_model=self.agent_protocols_by_model,
            agent_memory_by_player=self.agent_memory_by_player,
        )

    def save(self, path: Path) -> None:
        serialized = self.create_save().model_dump_json(indent=2)
        _atomic_write_text(path, serialized)

    @classmethod
    def from_save(
        cls,
        save: RunnerSave,
        *,
        models_by_spec: Mapping[str, AgentImplementation],
        board: BoardSpec = WAR1_BOARD_SPEC,
    ) -> GameSession:
        try:
            restored = restore_snapshot(save.snapshot)
        except (SnapshotRestoreError, InvalidRandomStreamStateError) as error:
            raise RunnerSaveError(f"Could not restore the saved game: {error}") from error
        if (
            not isinstance(restored.state, FullGameState)
            or restored.random_streams is None
        ):
            raise RunnerSaveError(
                "Saved game does not contain a restorable full game state."
            )
        return cls(
            state=restored.state,
            human_player_id=save.human_player_id,
            model_specs_by_player=save.model_specs_by_player,
            models_by_spec=models_by_spec,
            agent_memory_by_player=save.agent_memory_by_player,
            expected_protocols_by_model=save.agent_protocols_by_model,
            board=board,
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_runner_save(path: Path) -> RunnerSave:
    serialized = path.read_text(encoding="utf-8")
    try:
        return RunnerSave.model_validate_json(serialized)
    except ValidationError as error:
        raise RunnerSaveError(f"Invalid runner save {str(path)!r}: {error}") from error


def load_agent(model_spec: str) -> AgentImplementation:
    module_name, separator, factory_name = model_spec.partition(":")
    if not separator or not module_name.strip() or not factory_name.strip():
        raise AgentLoadError(
            f"Model {model_spec!r} must use the module:factory format."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise AgentLoadError(
            f"Could not import model factory {model_spec!r}: {error}"
        ) from error
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise AgentLoadError(
            f"Model factory {factory_name!r} is not callable in {module_name!r}."
        )
    candidate: object = factory()
    if not isinstance(candidate, AgentImplementation):
        raise AgentLoadError(
            f"Factory {model_spec!r} did not return an AgentImplementation."
        )
    return candidate


def load_agents(model_specs: Sequence[str]) -> dict[str, AgentImplementation]:
    return {
        model_spec: load_agent(model_spec)
        for model_spec in dict.fromkeys(model_specs)
    }


def model_specs_for_seats(
    player_count: int,
    human_seat: int,
    *,
    shared_model: str | None,
    seat_models: Sequence[str] = (),
) -> dict[str, str]:
    minimum, maximum = PLAYER_COUNT_RANGE
    if not minimum <= player_count <= maximum:
        raise RunnerConfigurationError(
            f"WAR 1 classic requires between {minimum} and {maximum} players."
        )
    if not 1 <= human_seat <= player_count:
        raise RunnerConfigurationError(
            f"Human seat must be between 1 and {player_count}."
        )
    if shared_model is not None and seat_models:
        raise RunnerConfigurationError(
            "Use either --model or --seat-model, not both."
        )

    ai_seats = set(range(1, player_count + 1)) - {human_seat}
    if shared_model is not None:
        if not shared_model.strip():
            raise RunnerConfigurationError("Model specification cannot be blank.")
        return {
            f"player-{seat}": shared_model.strip()
            for seat in sorted(ai_seats)
        }
    if not seat_models:
        raise RunnerConfigurationError(
            "Provide --model or assign every AI seat with --seat-model."
        )

    model_by_seat: dict[int, str] = {}
    for assignment in seat_models:
        seat_text, separator, model_spec = assignment.partition("=")
        if not separator or not model_spec.strip():
            raise RunnerConfigurationError(
                f"Seat model {assignment!r} must use the SEAT=module:factory format."
            )
        try:
            seat = int(seat_text)
        except ValueError as error:
            raise RunnerConfigurationError(
                f"Invalid seat number in assignment {assignment!r}."
            ) from error
        if seat not in ai_seats:
            raise RunnerConfigurationError(
                f"Seat-model assignments must target AI seats {sorted(ai_seats)}."
            )
        if seat in model_by_seat:
            raise RunnerConfigurationError(
                f"AI seat {seat} has more than one model assignment."
            )
        model_by_seat[seat] = model_spec.strip()

    if set(model_by_seat) != ai_seats:
        missing_seats = sorted(ai_seats - set(model_by_seat))
        raise RunnerConfigurationError(
            f"Assign a model to every AI seat; missing seats: {missing_seats}."
        )
    return {
        f"player-{seat}": model_by_seat[seat]
        for seat in sorted(model_by_seat)
    }


def _territory_label(
    territory_id: str,
    board: BoardSpec = WAR1_BOARD_SPEC,
) -> str:
    territory = next(
        (
            item
            for item in board.map.territories
            if item.territory_id == territory_id
        ),
        None,
    )
    if territory is None:
        return territory_id
    return f"{territory.canonical_name} ({territory_id})"


def format_action(
    action: Action,
    *,
    board: BoardSpec = WAR1_BOARD_SPEC,
) -> str:
    operation = action.operation
    if isinstance(operation, ClaimTerritoryOperation):
        return f"Claim {_territory_label(operation.territory_id, board)}"
    if isinstance(operation, PlaceArmiesOperation):
        placement = operation.placements[0]
        return (
            f"Place 1 army on {_territory_label(placement.territory_id, board)}"
        )
    if isinstance(operation, TradeCardsOperation):
        cards = ", ".join(
            _territory_label(card_id, board) for card_id in operation.card_ids
        )
        return f"Trade cards: {cards}"
    if isinstance(operation, DrawTerritoryCardOperation):
        return "Draw a territory card"
    if isinstance(operation, DiscardTerritoryCardOperation):
        return f"Discard {_territory_label(operation.card_id, board)}"
    if isinstance(operation, AttackOperation):
        return (
            f"Attack {_territory_label(operation.target_territory_id, board)} "
            f"from {_territory_label(operation.origin_territory_id, board)} "
            f"with {operation.dice_count} dice"
        )
    if isinstance(operation, OccupyTerritoryOperation):
        return (
            f"Occupy {_territory_label(operation.target_territory_id, board)} "
            f"with {operation.army_count} armies"
        )
    if isinstance(operation, MoveArmiesOperation):
        return (
            f"Move {operation.army_count} armies from "
            f"{_territory_label(operation.origin_territory_id, board)} to "
            f"{_territory_label(operation.target_territory_id, board)}"
        )
    if isinstance(operation, EndPhaseOperation):
        return f"End the {operation.phase_id} phase"
    if isinstance(operation, EndTurnOperation):
        return "End turn"
    return operation.action_type


def render_human_turn(
    state: FullGameState,
    human_player_id: str,
    actions: Sequence[Action],
    *,
    board: BoardSpec = WAR1_BOARD_SPEC,
) -> str:
    observation = observe_player(
        state,
        human_player_id,
        legal_actions=tuple(actions),
    )
    public = observation.public_state
    player_by_id = {player.player_id: player for player in public.players}
    seat_by_id = {
        player.public.player_id: index
        for index, player in enumerate(state.players, start=1)
    }
    lines = [
        f"Game {state.game_id} — turn {state.turn_number}, phase {state.phase_id}",
        "",
        "Players:",
    ]
    for player in public.players:
        status = "eliminated" if player.is_eliminated else "active"
        role = " (human)" if player.player_id == human_player_id else ""
        lines.append(
            f"  {seat_by_id[player.player_id]}. {player.player_id} "
            f"[{player.color}]{role}: {player.territory_count} territories, "
            f"{player.army_count} armies, {status}"
        )

    lines.extend(("", "Board:"))
    for territory in state.territories:
        owner = (
            player_by_id.get(territory.owner_player_id)
            if territory.owner_player_id is not None
            else None
        )
        owner_label = (
            f"{owner.player_id} [{owner.color}]" if owner is not None else "unowned"
        )
        lines.append(
            f"  {_territory_label(territory.territory_id, board)}: "
            f"{owner_label}, {territory.army_count} armies"
        )

    objective_id = observation.private_state.objective_id
    objective = next(
        (
            item
            for item in board.objectives
            if item.objective_id == objective_id
        ),
        None,
    )
    if objective is not None:
        lines.extend(
            (
                "",
                f"Your private objective ({objective.objective_id}): "
                f"{objective.description}",
            )
        )
    cards = observation.private_state.hand_card_ids
    card_labels = ", ".join(_territory_label(card_id, board) for card_id in cards)
    lines.append(f"Your private cards: {card_labels or '(none)'}")
    lines.extend(("", "Legal actions:"))
    lines.extend(
        f"  {index}. {format_action(action, board=board)}"
        for index, action in enumerate(actions, start=1)
    )
    return "\n".join(lines)


def _format_combat_outcome(outcome: ChanceOutcome) -> str:
    if isinstance(outcome.outcome, dict):
        attacker_faces = outcome.outcome.get("attacker_faces", "?")
        defender_faces = outcome.outcome.get("defender_faces", "?")
        return (
            f"Combat resolved — attacker dice: {attacker_faces}; "
            f"defender dice: {defender_faces}"
        )
    return f"Combat resolved — outcome: {outcome.outcome}"


def _render_game_over(
    state: FullGameState,
    *,
    board: BoardSpec = WAR1_BOARD_SPEC,
) -> str:
    public = to_public_game_state(state)
    if public.winner_player_id is None:
        return "Game finished without a winner."
    objective_id = (
        public.winner_objective_fallback_id or public.winner_objective_id
    )
    objective = next(
        (
            item
            for item in board.objectives
            if item.objective_id == objective_id
        ),
        None,
    )
    if objective is None:
        return f"Winner: {public.winner_player_id}."
    return (
        f"Winner: {public.winner_player_id}. Revealed objective: "
        f"{objective.description}"
    )


def _run_session(
    session: GameSession,
    *,
    save_path: Path | None,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> bool:
    if save_path is not None:
        session.save(save_path)

    while not session.state.is_finished:
        if session.state.pending_combat is not None:
            outcome = session.resolve_pending_combat()
            output_fn(_format_combat_outcome(outcome))
        else:
            player_id = session.decision_player_id
            if player_id is None:
                raise RunnerConfigurationError(
                    "A nonterminal game has no decision player."
                )
            if player_id == session.human_player_id:
                actions = session.available_actions()
                if not actions:
                    raise RunnerConfigurationError(
                        f"No legal actions are available in phase "
                        f"{session.state.phase_id!r}."
                    )
                output_fn(
                    render_human_turn(
                        session.state,
                        session.human_player_id,
                        actions,
                        board=session.board,
                    )
                )
                try:
                    selection = input_fn(
                        "Choose an action number, or q to pause: "
                    ).strip()
                except EOFError:
                    output_fn("Input ended; game paused.")
                    return False
                if selection.casefold() in {"q", "quit", "exit"}:
                    if save_path is None:
                        output_fn(
                            "Game paused without a save. Start with --save PATH "
                            "to resume later."
                        )
                    else:
                        output_fn(f"Game paused and saved to {save_path}.")
                    return False
                try:
                    action_index = int(selection)
                except ValueError:
                    output_fn("Enter a listed action number or q.")
                    continue
                if not 1 <= action_index <= len(actions):
                    output_fn(
                        f"Choose a number between 1 and {len(actions)}."
                    )
                    continue
                action = actions[action_index - 1]
                session.apply_human_action(action)
                output_fn(f"You chose: {format_action(action, board=session.board)}")
            else:
                action, elapsed_ms = session.play_agent_turn()
                model_spec = session.model_specs_by_player[player_id]
                agent_id = session.agent_protocols_by_model[model_spec].agent_id
                output_fn(
                    f"{agent_id} ({player_id}) chose: "
                    f"{format_action(action, board=session.board)} "
                    f"({elapsed_ms:.1f} ms)"
                )
        if save_path is not None:
            session.save(save_path)

    output_fn(_render_game_over(session.state, board=session.board))
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play WAR 1 classic against AgentProtocol-compatible models."
    )
    parser.add_argument("--players", type=int, help="Number of seats (3-6).")
    parser.add_argument("--human-seat", type=int, help="Human seat, numbered from 1.")
    parser.add_argument(
        "--model",
        help="Agent factory module:callable; assign it to every AI seat.",
    )
    parser.add_argument(
        "--seat-model",
        action="append",
        default=[],
        metavar="SEAT=MODULE:FACTORY",
        help="Assign a model to one AI seat; repeat for each AI seat.",
    )
    parser.add_argument("--seed", type=int, help="Non-negative game root seed.")
    parser.add_argument("--game-id", help="Optional ID for a new game.")
    parser.add_argument("--save", type=Path, help="Save after each state transition.")
    parser.add_argument("--resume", type=Path, help="Resume a saved game.")
    return parser


def _session_from_args(
    args: argparse.Namespace,
) -> tuple[GameSession, Path | None]:
    if args.resume is not None:
        if any(
            value is not None
            for value in (args.players, args.human_seat, args.seed, args.game_id)
        ) or args.model is not None or args.seat_model:
            raise RunnerConfigurationError(
                "When using --resume, player, seed, and model settings come "
                "from the save file."
            )
        save = load_runner_save(args.resume)
        models = load_agents(tuple(save.model_specs_by_player.values()))
        session = GameSession.from_save(save, models_by_spec=models)
        return session, args.save or args.resume

    if args.players is None:
        raise RunnerConfigurationError("--players is required for a new game.")
    if args.human_seat is None:
        raise RunnerConfigurationError("--human-seat is required for a new game.")
    if args.seed is None:
        raise RunnerConfigurationError("--seed is required for a new game.")
    if args.seed < 0:
        raise RunnerConfigurationError("--seed must be non-negative.")
    model_specs = model_specs_for_seats(
        args.players,
        args.human_seat,
        shared_model=args.model,
        seat_models=args.seat_model,
    )
    models = load_agents(tuple(model_specs.values()))
    players = tuple(
        SetupPlayer(
            player_id=f"player-{seat}",
            color=PLAYER_COLORS[seat - 1],
        )
        for seat in range(1, args.players + 1)
    )
    setup = prepare_game(
        args.game_id or f"war-{uuid4().hex}",
        players,
        args.seed,
    )
    session = GameSession(
        state=setup.state,
        human_player_id=f"player-{args.human_seat}",
        model_specs_by_player=model_specs,
        models_by_spec=models,
    )
    return session, args.save


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    read_input = input if input_fn is None else input_fn
    try:
        session, save_path = _session_from_args(args)
        _run_session(
            session,
            save_path=save_path,
            input_fn=read_input,
            output_fn=output_fn,
        )
    except (
        AgentLoadError,
        AgentCompatibilityError,
        AgentDecisionError,
        RunnerConfigurationError,
        RunnerSaveError,
        InvalidSetupError,
        SnapshotRestoreError,
        InvalidRandomStreamStateError,
        OSError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
