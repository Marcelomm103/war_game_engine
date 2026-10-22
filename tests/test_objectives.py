from collections.abc import Mapping

import pytest

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    ArmyPlacement,
    DeckState,
    FullGameState,
    FullPlayerState,
    PendingObjectiveCheck,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from war_game_engine.engine import (
    VictoryAdjudicationRequiredError,
    apply_action,
    resolve_pending_objective_check,
)
from war_game_engine.objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    InvalidObjectiveStateError,
    TerritoryCountCondition,
    effective_objective_id,
    objective_is_satisfied,
    objective_winners,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.state import to_public_game_state
from war_game_engine.versioning import ACTION_SCHEMA_VERSION


def _objective_state(
    *,
    objective_ids: tuple[str, ...] = (
        "W1-OBJ-13",
        "W1-OBJ-14",
        "W1-OBJ-07",
    ),
    colors: tuple[str, ...] = ("blue", "red", "green"),
    owner_by_territory: Mapping[str, str | None] | None = None,
    army_by_territory: dict[str, int] | None = None,
    objective_fallbacks: dict[str, str] | None = None,
) -> FullGameState:
    if len(objective_ids) != len(colors):
        raise ValueError("Objective fixtures require one unique objective per player.")
    player_ids = tuple(f"player-{index + 1}" for index in range(len(colors)))
    owners = owner_by_territory or {
        territory.territory_id: player_ids[index % len(player_ids)]
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    armies = army_by_territory or {}
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=owners.get(territory.territory_id),
            army_count=(
                0
                if owners.get(territory.territory_id) is None
                else armies.get(territory.territory_id, 1)
            ),
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    )
    players: list[FullPlayerState] = []
    for index, player_id in enumerate(player_ids):
        owned = tuple(
            territory
            for territory in territories
            if territory.owner_player_id == player_id
        )
        players.append(
            FullPlayerState(
                public=PublicPlayerState(
                    player_id=player_id,
                    color=colors[index],
                    territory_count=len(owned),
                    army_count=sum(territory.army_count for territory in owned),
                    is_eliminated=not owned,
                ),
                private=PlayerPrivateState(
                    player_id=player_id,
                    objective_id=objective_ids[index],
                    objective_fallback_id=(
                        objective_fallbacks.get(player_id)
                        if objective_fallbacks is not None
                        else None
                    ),
                    hand_card_ids=(),
                ),
            )
        )
    if not any(
        territory.owner_player_id == player_ids[0] for territory in territories
    ):
        raise ValueError("The first objective-test player must remain active.")
    return FullGameState(
        contract_type="full_game_state",
        game_id="objective-test",
        ruleset_id=WAR1_BOARD_SPEC.board_id,
        rules_version="1.0.0",
        state_version=0,
        phase_id="attack",
        turn_number=2,
        active_player_id=player_ids[0],
        turn_player_id=player_ids[0],
        decision_player_id=player_ids[0],
        initial_placement_order=player_ids,
        players=tuple(players),
        territories=territories,
        reinforcement_pools=(),
        deck=DeckState(
            draw_pile_card_ids=(),
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(42).snapshot(),
        is_finished=False,
    )


def test_typed_catalog_covers_every_canonical_objective() -> None:
    assert set(OBJECTIVE_CONDITIONS) == set(WAR1_BOARD_SPEC.objective_deck.card_ids)
    assert isinstance(OBJECTIVE_CONDITIONS["W1-OBJ-01"], EliminateColorCondition)
    assert isinstance(OBJECTIVE_CONDITIONS["W1-OBJ-07"], ControlContinentsCondition)
    assert isinstance(OBJECTIVE_CONDITIONS["W1-OBJ-13"], TerritoryCountCondition)


@pytest.mark.parametrize(
    ("objective_id", "target_color"),
    tuple(ELIMINATION_OBJECTIVE_COLORS.items()),
)
def test_each_elimination_objective_requires_its_target_color_to_be_eliminated(
    objective_id: str,
    target_color: str,
) -> None:
    all_colors = ("white", "red", "black", "blue", "yellow", "green")
    holder_color = next(color for color in all_colors if color != target_color)
    colors = (holder_color, target_color) + tuple(
        color for color in all_colors if color not in {holder_color, target_color}
    )
    player_ids = tuple(f"player-{index + 1}" for index in range(len(colors)))
    objective_ids = (
        objective_id,
        "W1-OBJ-07",
        "W1-OBJ-08",
        "W1-OBJ-09",
        "W1-OBJ-10",
        "W1-OBJ-13",
    )
    owners = {
        territory.territory_id: player_ids[0]
        for territory in WAR1_BOARD_SPEC.map.territories
    }
    state = _objective_state(
        objective_ids=objective_ids,
        colors=colors,
        owner_by_territory=owners,
    )

    assert objective_is_satisfied(state, state.players[0]) is True


def test_territory_count_and_garrison_objectives_use_exact_thresholds() -> None:
    player_ids = ("player-1", "player-2", "player-3")
    owners = {
        territory.territory_id: (
            player_ids[0] if index < 24 else player_ids[1]
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    territory_state = _objective_state(
        objective_ids=("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-07"),
        owner_by_territory=owners,
    )
    assert objective_is_satisfied(territory_state, territory_state.players[0])

    owners["T24"] = "player-2"
    below_24 = _objective_state(
        objective_ids=("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-07"),
        owner_by_territory=owners,
    )
    assert not objective_is_satisfied(below_24, below_24.players[0])

    garrison_owners = {
        territory.territory_id: (
            player_ids[0] if index < 18 else player_ids[1]
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    armies = {f"T{index:02d}": 2 for index in range(1, 18)}
    armies["T18"] = 1
    garrison_state = _objective_state(
        objective_ids=("W1-OBJ-14", "W1-OBJ-13", "W1-OBJ-07"),
        owner_by_territory=garrison_owners,
        army_by_territory=armies,
    )
    assert not objective_is_satisfied(garrison_state, garrison_state.players[0])
    armies["T18"] = 2
    completed_garrison = _objective_state(
        objective_ids=("W1-OBJ-14", "W1-OBJ-13", "W1-OBJ-07"),
        owner_by_territory=garrison_owners,
        army_by_territory=armies,
    )
    assert objective_is_satisfied(completed_garrison, completed_garrison.players[0])


def test_continent_objectives_require_all_named_and_additional_continents() -> None:
    player_ids = ("player-1", "player-2", "player-3")
    north_america_and_oceania = {
        territory.territory_id: (
            player_ids[0]
            if territory.continent_id in {"north_america", "oceania"}
            else player_ids[1]
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    }
    two_continents = _objective_state(
        objective_ids=("W1-OBJ-07", "W1-OBJ-14", "W1-OBJ-13"),
        owner_by_territory=north_america_and_oceania,
    )
    assert objective_is_satisfied(two_continents, two_continents.players[0])
    north_america_and_oceania["T01"] = "player-2"
    missing_one = _objective_state(
        objective_ids=("W1-OBJ-07", "W1-OBJ-14", "W1-OBJ-13"),
        owner_by_territory=north_america_and_oceania,
    )
    assert not objective_is_satisfied(missing_one, missing_one.players[0])

    required_and_additional = {
        territory.territory_id: (
            player_ids[0]
            if territory.continent_id in {"europe", "south_america"}
            else player_ids[1]
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    }
    required_only = _objective_state(
        objective_ids=("W1-OBJ-11", "W1-OBJ-14", "W1-OBJ-13"),
        owner_by_territory=required_and_additional,
    )
    assert not objective_is_satisfied(required_only, required_only.players[0])
    for territory in WAR1_BOARD_SPEC.map.territories:
        if territory.continent_id == "africa":
            required_and_additional[territory.territory_id] = player_ids[0]
    with_additional = _objective_state(
        objective_ids=("W1-OBJ-11", "W1-OBJ-14", "W1-OBJ-13"),
        owner_by_territory=required_and_additional,
    )
    assert objective_is_satisfied(with_additional, with_additional.players[0])


@pytest.mark.parametrize(
    ("objective_id", "required_continents", "additional_continents"),
    (
        ("W1-OBJ-08", ("asia", "south_america"), ()),
        ("W1-OBJ-09", ("asia", "africa"), ()),
        ("W1-OBJ-10", ("north_america", "africa"), ()),
        ("W1-OBJ-12", ("europe", "oceania"), ("asia",)),
    ),
)
def test_remaining_continent_objectives_match_the_physical_catalog(
    objective_id: str,
    required_continents: tuple[str, ...],
    additional_continents: tuple[str, ...],
) -> None:
    player_ids = ("player-1", "player-2", "player-3")
    controlled_continents = set(required_continents + additional_continents)
    owners = {
        territory.territory_id: (
            player_ids[0]
            if territory.continent_id in controlled_continents
            else player_ids[1]
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    }
    state = _objective_state(
        objective_ids=(objective_id, "W1-OBJ-14", "W1-OBJ-13"),
        owner_by_territory=owners,
    )

    assert objective_is_satisfied(state, state.players[0])


def test_self_target_uses_card_fallback_and_missing_target_is_explicit() -> None:
    owners = {
        territory.territory_id: (
            "player-1" if index < 24 else "player-2"
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    self_target = _objective_state(
        objective_ids=("W1-OBJ-01", "W1-OBJ-14", "W1-OBJ-07"),
        colors=("red", "blue", "green"),
        owner_by_territory=owners,
        objective_fallbacks={"player-1": FALLBACK_OBJECTIVE_ID},
    )
    assert effective_objective_id(self_target.players[0]) == FALLBACK_OBJECTIVE_ID
    assert objective_is_satisfied(self_target, self_target.players[0])
    finished_data = self_target.model_dump(mode="python")
    finished_data.update(
        {
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "winner_player_id": "player-1",
            "is_finished": True,
        }
    )
    public_winner = to_public_game_state(FullGameState.model_validate(finished_data))
    assert public_winner.winner_objective_id == "W1-OBJ-01"
    assert public_winner.winner_objective_fallback_id == FALLBACK_OBJECTIVE_ID

    without_fallback = _objective_state(
        objective_ids=("W1-OBJ-01", "W1-OBJ-14", "W1-OBJ-07"),
        colors=("red", "blue", "green"),
        owner_by_territory=owners,
    )
    with pytest.raises(InvalidObjectiveStateError, match="Self-targeting objective"):
        objective_is_satisfied(without_fallback, without_fallback.players[0])

    missing_target = _objective_state(
        objective_ids=("W1-OBJ-01", "W1-OBJ-14", "W1-OBJ-07"),
        colors=("blue", "green", "white"),
    )
    with pytest.raises(InvalidObjectiveStateError, match="targets absent color"):
        objective_is_satisfied(missing_target, missing_target.players[0])


def test_unknown_objective_id_is_rejected_instead_of_silently_losing() -> None:
    state = _objective_state()
    state_data = state.model_dump(mode="python")
    players = list(state.players)
    first_player = players[0]
    players[0] = FullPlayerState(
        public=first_player.public,
        private=PlayerPrivateState.model_validate(
            {
                **first_player.private.model_dump(),
                "objective_id": "unknown-objective",
            }
        ),
    )
    state_data["players"] = tuple(players)
    state = FullGameState.model_validate(state_data)

    with pytest.raises(InvalidObjectiveStateError, match="Unknown objective ID"):
        objective_is_satisfied(state, state.players[0])


def test_objective_winners_lists_all_simultaneous_winners_without_priority() -> None:
    players = ("player-1", "player-2", "player-3")
    owners = {
        territory.territory_id: (
            players[0] if index < 24 else players[1]
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    state = _objective_state(
        objective_ids=("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-07"),
        owner_by_territory=owners,
        army_by_territory={
            f"T{index:02d}": 2 for index in range(25, 43)
        },
    )

    assert objective_winners(state) == ("player-1", "player-2")


def test_continent_evaluation_rejects_incomplete_territory_state() -> None:
    state = _objective_state(objective_ids=("W1-OBJ-07", "W1-OBJ-14", "W1-OBJ-13"))
    state_data = state.model_dump(mode="python")
    state_data["territories"] = state.territories[:-1]
    state_data["players"] = tuple(
        FullPlayerState(
            public=PublicPlayerState.model_validate(
                {
                    **player.public.model_dump(),
                    "territory_count": sum(
                        territory.owner_player_id == player.public.player_id
                        for territory in state.territories[:-1]
                    ),
                    "army_count": sum(
                        territory.army_count
                        for territory in state.territories[:-1]
                        if territory.owner_player_id == player.public.player_id
                    ),
                    "is_eliminated": not any(
                        territory.owner_player_id == player.public.player_id
                        for territory in state.territories[:-1]
                    ),
                }
            ),
            private=player.private,
        )
        for player in state.players
    )
    partial_state = FullGameState.model_validate(state_data)

    with pytest.raises(InvalidObjectiveStateError, match="complete territory state"):
        objective_is_satisfied(partial_state, partial_state.players[0])


def test_objective_check_finishes_immediately_after_one_army_placement() -> None:
    player_ids = ("player-1", "player-2", "player-3")
    owners = {
        territory.territory_id: (
            player_ids[0] if index < 18 else (
                player_ids[1] if index < 30 else player_ids[2]
            )
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    armies = {f"T{index:02d}": 2 for index in range(1, 18)}
    armies["T18"] = 1
    state = _objective_state(
        objective_ids=("W1-OBJ-14", "W1-OBJ-13", "W1-OBJ-07"),
        owner_by_territory=owners,
        army_by_territory=armies,
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "initial_placement",
            "turn_number": 1,
            "reinforcement_pools": (
                ReinforcementPool(
                    pool_id="player-1:initial",
                    player_id="player-1",
                    origin="territory_count",
                    remaining_armies=1,
                ),
                ReinforcementPool(
                    pool_id="player-2:initial",
                    player_id="player-2",
                    origin="territory_count",
                    remaining_armies=1,
                ),
                ReinforcementPool(
                    pool_id="player-3:initial",
                    player_id="player-3",
                    origin="territory_count",
                    remaining_armies=1,
                ),
            ),
        }
    )
    state = FullGameState.model_validate(state_data)
    action = Action(
        contract_type="action",
        action_id="winning-placement",
        actor_player_id="player-1",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=PlaceArmiesOperation(
            action_type="place_armies",
            placements=(
                ArmyPlacement(
                    territory_id="T18",
                    army_count=1,
                    source_pool_id="player-1:initial",
                ),
            ),
            placement_phase="initial",
        ),
    )

    result = apply_action(state, action)

    assert result.phase_id == "finished"
    assert result.is_finished is True
    assert result.winner_player_id == "player-1"
    assert result.players[0].public.territory_count == 18
    assert result.territories[17].army_count == 2
    assert result.reinforcement_pools == ()
    assert result.decision_player_id is None
    public_state = to_public_game_state(result)
    assert public_state.winner_player_id == "player-1"
    assert public_state.winner_objective_id == "W1-OBJ-14"
    assert public_state.winner_objective_fallback_id is None


def test_unfinished_public_state_does_not_reveal_objectives() -> None:
    state = _objective_state()

    public_state = to_public_game_state(state)

    assert public_state.winner_player_id is None
    assert public_state.winner_objective_id is None
    assert public_state.winner_objective_fallback_id is None


def test_simultaneous_objectives_raise_adjudication_without_choosing_a_winner() -> None:
    player_ids = ("player-1", "player-2", "player-3")
    owners = {
        territory.territory_id: (
            player_ids[0] if index < 24 else player_ids[1]
        )
        for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
    }
    armies = {f"T{index:02d}": 2 for index in range(25, 43)}
    state = _objective_state(
        objective_ids=("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-07"),
        owner_by_territory=owners,
        army_by_territory=armies,
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "objective_check",
            "pending_objective_check": PendingObjectiveCheck(
                contract_type="pending_objective_check",
                actor_player_id="player-1",
                territory_id="T01",
                resume_phase_id="attack",
                resume_decision_player_id="player-1",
            ),
        }
    )
    state = FullGameState.model_validate(state_data)

    with pytest.raises(VictoryAdjudicationRequiredError) as error:
        resolve_pending_objective_check(state)

    assert error.value.winner_player_ids == ("player-1", "player-2")
    assert error.value.state == state
    assert state.winner_player_id is None
