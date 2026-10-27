from __future__ import annotations

from pathlib import Path

import pytest
from numpy.random import Generator
from pydantic import JsonValue

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    AgentProtocol,
    FullGameState,
    PlayerObservation,
)
from war_game_engine.runner import (
    AGENT_PROTOCOL_VERSION,
    AgentCompatibilityError,
    AgentDecision,
    AgentDecisionError,
    GameSession,
    RunnerConfigurationError,
    load_agent,
    load_runner_save,
    main,
    model_specs_for_seats,
    render_human_turn,
)
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.versioning import (
    ACTION_ENCODING_VERSION,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_ENCODING_VERSION,
    OBSERVATION_SCHEMA_VERSION,
)

PLAYER_SPECS = (
    SetupPlayer(player_id="player-1", color="white"),
    SetupPlayer(player_id="player-2", color="red"),
    SetupPlayer(player_id="player-3", color="blue"),
)
MODEL_SPEC = "test_runner:create_test_agent"


def _protocol(
    *,
    agent_id: str = "test-agent",
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION,
) -> AgentProtocol:
    return AgentProtocol(
        contract_type="agent_protocol",
        protocol_version=AGENT_PROTOCOL_VERSION,
        agent_id=agent_id,
        supported_ruleset_ids=(WAR1_BOARD_SPEC.board_id,),
        observation_schema_version=observation_schema_version,
        observation_encoding_version=OBSERVATION_ENCODING_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        action_encoding_version=ACTION_ENCODING_VERSION,
        memory_scope="per_player",
        rng_scope="private_per_player",
        decision_budget_ms=1_000,
    )


class FirstLegalAgent:
    def __init__(self, protocol: AgentProtocol | None = None) -> None:
        self.protocol = protocol or _protocol()

    def decide(
        self,
        observation: PlayerObservation,
        *,
        rng: Generator,
        memory: JsonValue | None,
    ) -> AgentDecision:
        del rng
        previous_count = 0
        if isinstance(memory, dict):
            count = memory.get("count")
            if isinstance(count, int):
                previous_count = count
        return AgentDecision(
            action=observation.legal_actions[0],
            memory={"count": previous_count + 1},
        )


class CapturingAgent(FirstLegalAgent):
    def __init__(self) -> None:
        super().__init__()
        self.last_observation: PlayerObservation | None = None

    def decide(
        self,
        observation: PlayerObservation,
        *,
        rng: Generator,
        memory: JsonValue | None,
    ) -> AgentDecision:
        self.last_observation = observation
        return super().decide(observation, rng=rng, memory=memory)


def create_test_agent() -> FirstLegalAgent:
    return FirstLegalAgent()


def _initial_state(seed: int = 23) -> FullGameState:
    return prepare_game("runner-test", PLAYER_SPECS, seed).state


def _session(
    *,
    human_player_id: str | None = None,
    agent: FirstLegalAgent | None = None,
) -> GameSession:
    state = _initial_state()
    if human_player_id is None:
        decision_player_id = state.decision_player_id
        assert decision_player_id is not None
        human_player_id = next(
            player.public.player_id
            for player in state.players
            if player.public.player_id != decision_player_id
        )
    ai_player_ids = {
        player.public.player_id
        for player in state.players
        if player.public.player_id != human_player_id
    }
    return GameSession(
        state=state,
        human_player_id=human_player_id,
        model_specs_by_player={
            player_id: MODEL_SPEC for player_id in ai_player_ids
        },
        models_by_spec={MODEL_SPEC: agent or FirstLegalAgent()},
    )


def test_shared_and_per_seat_model_assignments() -> None:
    assert model_specs_for_seats(
        4,
        2,
        shared_model="pkg.agents:make_agent",
    ) == {
        "player-1": "pkg.agents:make_agent",
        "player-3": "pkg.agents:make_agent",
        "player-4": "pkg.agents:make_agent",
    }
    assert model_specs_for_seats(
        4,
        2,
        shared_model=None,
        seat_models=(
            "1=pkg.agents:make_first",
            "3=pkg.agents:make_second",
            "4=pkg.agents:make_first",
        ),
    ) == {
        "player-1": "pkg.agents:make_first",
        "player-3": "pkg.agents:make_second",
        "player-4": "pkg.agents:make_first",
    }


def test_model_assignments_reject_missing_duplicate_or_human_seats() -> None:
    with pytest.raises(RunnerConfigurationError, match="missing seats"):
        model_specs_for_seats(
            4,
            2,
            shared_model=None,
            seat_models=("1=pkg.agents:make_agent",),
        )
    with pytest.raises(RunnerConfigurationError, match="more than one"):
        model_specs_for_seats(
            3,
            2,
            shared_model=None,
            seat_models=(
                "1=pkg.agents:make_first",
                "1=pkg.agents:make_second",
                "3=pkg.agents:make_agent",
            ),
        )
    with pytest.raises(RunnerConfigurationError, match="AI seats"):
        model_specs_for_seats(
            3,
            2,
            shared_model=None,
            seat_models=(
                "1=pkg.agents:make_agent",
                "2=pkg.agents:make_agent",
            ),
        )


def test_agent_factory_is_loaded_from_module_reference() -> None:
    assert isinstance(load_agent(f"{__name__}:create_test_agent"), FirstLegalAgent)


def test_agent_receives_only_its_player_observation() -> None:
    state = _initial_state()
    decision_player_id = state.decision_player_id
    assert decision_player_id is not None
    human_player_id = next(
        player.public.player_id
        for player in state.players
        if player.public.player_id != decision_player_id
    )
    agent = CapturingAgent()
    session = _session(human_player_id=human_player_id, agent=agent)

    session.play_agent_turn()

    observation = agent.last_observation
    assert observation is not None
    assert observation.player_id == decision_player_id
    assert observation.private_state.player_id == decision_player_id
    assert not hasattr(observation, "deck")
    assert not hasattr(observation, "random_streams_state")
    assert all(not hasattr(player, "private") for player in observation.public_state.players)
    other_private_ids = {
        player.private.objective_id
        for player in state.players
        if player.public.player_id != decision_player_id
    }
    observation_json = observation.model_dump_json()
    assert all(
        private_id is None or private_id not in observation_json
        for private_id in other_private_ids
    )


def test_human_view_includes_only_its_private_objective_and_cards() -> None:
    state = _initial_state()
    human_player_id = state.decision_player_id
    assert human_player_id is not None
    opponent_id = next(
        player.public.player_id
        for player in state.players
        if player.public.player_id != human_player_id
    )
    state_data = state.model_dump(mode="python")
    draw_pile = list(state.deck.draw_pile_card_ids)
    human_card, opponent_card = draw_pile[:2]
    state_data["deck"]["draw_pile_card_ids"] = tuple(draw_pile[2:])
    for player in state_data["players"]:
        player_id = player["public"]["player_id"]
        if player_id == human_player_id:
            player["private"]["hand_card_ids"] = (human_card,)
        elif player_id == opponent_id:
            player["private"]["hand_card_ids"] = (opponent_card,)
    state_with_cards = FullGameState.model_validate(state_data)
    actions = GameSession(
        state=state_with_cards,
        human_player_id=human_player_id,
        model_specs_by_player={
            player.public.player_id: MODEL_SPEC
            for player in state.players
            if player.public.player_id != human_player_id
        },
        models_by_spec={MODEL_SPEC: FirstLegalAgent()},
    ).available_actions()

    rendered = render_human_turn(
        state_with_cards,
        human_player_id,
        actions,
    )
    human_player = next(
        player
        for player in state_with_cards.players
        if player.public.player_id == human_player_id
    )
    opponent = next(
        player
        for player in state_with_cards.players
        if player.public.player_id == opponent_id
    )
    opponent_objective_id = opponent.private.objective_id
    objective = next(
        item
        for item in WAR1_BOARD_SPEC.objectives
        if item.objective_id == human_player.private.objective_id
    )
    private_cards_line = next(
        line for line in rendered.splitlines() if line.startswith("Your private cards:")
    )

    assert objective.description in rendered
    assert human_card in private_cards_line
    assert opponent_card not in private_cards_line
    assert opponent_objective_id is not None
    assert opponent_objective_id not in rendered


def test_agent_protocol_must_match_current_observation_schema() -> None:
    state = _initial_state()
    human_player_id = next(
        player.public.player_id
        for player in state.players
        if player.public.player_id != state.decision_player_id
    )
    ai_player_ids = {
        player.public.player_id
        for player in state.players
        if player.public.player_id != human_player_id
    }

    with pytest.raises(AgentCompatibilityError, match="observation schema"):
        GameSession(
            state=state,
            human_player_id=human_player_id,
            model_specs_by_player={
                player_id: MODEL_SPEC for player_id in ai_player_ids
            },
            models_by_spec={
                MODEL_SPEC: FirstLegalAgent(
                    protocol=_protocol(observation_schema_version="1.0.0")
                )
            },
        )


def test_save_and_resume_preserves_state_rng_and_agent_memory(
    tmp_path: Path,
) -> None:
    session = _session()
    decision_player_id = session.decision_player_id
    assert decision_player_id != session.human_player_id
    session.play_agent_turn()
    path = tmp_path / "nested" / "game.json"
    session.save(path)

    loaded_save = load_runner_save(path)
    restored = GameSession.from_save(
        loaded_save,
        models_by_spec={MODEL_SPEC: FirstLegalAgent()},
    )
    assert restored.state == session.state
    assert restored.agent_memory_by_player == session.agent_memory_by_player

    expected_action, _ = session.play_agent_turn()
    resumed_action, _ = restored.play_agent_turn()

    assert resumed_action == expected_action
    assert restored.state == session.state
    assert restored.agent_memory_by_player == session.agent_memory_by_player


def test_cli_saves_initial_state_and_can_pause(tmp_path: Path) -> None:
    initial_state = _initial_state()
    human_seat = next(
        index
        for index, player in enumerate(initial_state.players, start=1)
        if player.public.player_id == initial_state.decision_player_id
    )
    save_path = tmp_path / "paused-game.json"
    output: list[str] = []

    result = main(
        [
            "--players",
            "3",
            "--human-seat",
            str(human_seat),
            "--seed",
            "23",
            "--model",
            f"{__name__}:create_test_agent",
            "--save",
            str(save_path),
        ],
        input_fn=lambda _prompt: "q",
        output_fn=output.append,
    )

    assert result == 0
    assert save_path.is_file()
    save = load_runner_save(save_path)
    assert save.human_player_id == f"player-{human_seat}"
    assert any("Your private objective" in line for line in output)


def test_human_action_must_be_currently_legal() -> None:
    state = _initial_state()
    session = _session(human_player_id=state.decision_player_id)
    action = session.available_actions()[0]

    session.apply_human_action(action)

    with pytest.raises(
        AgentDecisionError,
        match="not in the current legal-action list",
    ):
        session.apply_human_action(action)
