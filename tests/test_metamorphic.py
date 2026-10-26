from collections.abc import Mapping
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    ActionOperation,
    Adjacency,
    AttackOperation,
    BoardSpec,
    DeckState,
    FullGameState,
    FullPlayerState,
    MapSpec,
    MoveArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    TerritoryState,
)
from war_game_engine.engine import (
    InvalidActionError,
    _adjacent,
    apply_action,
    legal_actions,
    validate_action,
)
from war_game_engine.objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    objective_winners,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.state import observe_player, to_public_game_state
from war_game_engine.versioning import ACTION_SCHEMA_VERSION

PLAYER_IDS = ("player-1", "player-2", "player-3")
PLAYER_COLORS = ("blue", "red", "green")
PHYSICAL_COLORS = ("white", "red", "black", "blue", "yellow", "green")
TERRITORY_IDS = tuple(
    territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
)
ADJACENT_PAIRS = tuple(
    (adjacency.first_territory_id, adjacency.second_territory_id)
    for adjacency in WAR1_BOARD_SPEC.map.adjacencies
)
CARD_IDS = WAR1_BOARD_SPEC.territory_deck.card_ids
OBJECTIVE_FOR_COLOR: dict[str, str] = {
    color: objective_id
    for objective_id, color in ELIMINATION_OBJECTIVE_COLORS.items()
}


def _objective_state(target_eliminated: bool) -> FullGameState:
    owners: dict[str, str] = {}
    for index, territory_id in enumerate(TERRITORY_IDS):
        if index < 24:
            owner_player_id = PLAYER_IDS[0]
        elif index < 41:
            owner_player_id = PLAYER_IDS[2]
        else:
            owner_player_id = (
                PLAYER_IDS[0] if target_eliminated else PLAYER_IDS[1]
            )
        owners[territory_id] = owner_player_id

    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=owners[territory_id],
            army_count=1,
        )
        for territory_id in TERRITORY_IDS
    )
    eliminated_by = PLAYER_IDS[0] if target_eliminated else None
    players: list[FullPlayerState] = []
    for player_id, color, objective_id in zip(
        PLAYER_IDS,
        PLAYER_COLORS,
        ("W1-OBJ-01", "W1-OBJ-13", "W1-OBJ-14"),
        strict=True,
    ):
        owned_territories = tuple(
            territory
            for territory in territories
            if territory.owner_player_id == player_id
        )
        players.append(
            FullPlayerState(
                public=PublicPlayerState(
                    player_id=player_id,
                    color=color,
                    territory_count=len(owned_territories),
                    army_count=sum(
                        territory.army_count for territory in owned_territories
                    ),
                    is_eliminated=not owned_territories,
                    eliminated_by_player_id=(
                        eliminated_by if player_id == PLAYER_IDS[1] else None
                    ),
                ),
                private=PlayerPrivateState(
                    player_id=player_id,
                    objective_id=objective_id,
                    hand_card_ids=(),
                ),
            )
        )
    return FullGameState(
        contract_type="full_game_state",
        game_id="metamorphic-objectives",
        ruleset_id=WAR1_BOARD_SPEC.board_id,
        rules_version="1.0.0",
        state_version=0,
        phase_id="attack",
        turn_number=2,
        active_player_id=PLAYER_IDS[0],
        turn_player_id=PLAYER_IDS[0],
        decision_player_id=PLAYER_IDS[0],
        initial_placement_order=PLAYER_IDS,
        players=tuple(players),
        territories=territories,
        deck=DeckState(
            draw_pile_card_ids=CARD_IDS,
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(110).snapshot(),
        is_finished=False,
    )


def _remapped_objective_id(
    objective_id: str | None,
    color_mapping: Mapping[str, str],
) -> str | None:
    if objective_id is None:
        return None
    target_color = ELIMINATION_OBJECTIVE_COLORS.get(objective_id)
    if target_color is None:
        return objective_id
    return OBJECTIVE_FOR_COLOR[color_mapping[target_color]]


def _relabel_players_and_colors(
    state: FullGameState,
    player_mapping: Mapping[str, str],
    color_mapping: Mapping[str, str],
) -> FullGameState:
    players: list[FullPlayerState] = []
    for player in state.players:
        public_data = player.public.model_dump()
        public_data["player_id"] = player_mapping[player.public.player_id]
        public_data["color"] = color_mapping[player.public.color]
        public_data["eliminated_by_player_id"] = (
            player_mapping[player.public.eliminated_by_player_id]
            if player.public.eliminated_by_player_id is not None
            else None
        )
        private_data = player.private.model_dump()
        private_data["player_id"] = player_mapping[player.private.player_id]
        private_data["objective_id"] = _remapped_objective_id(
            player.private.objective_id,
            color_mapping,
        )
        players.append(
            FullPlayerState(
                public=PublicPlayerState.model_validate(public_data),
                private=PlayerPrivateState.model_validate(private_data),
            )
        )

    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                player_mapping[territory.owner_player_id]
                if territory.owner_player_id is not None
                else None
            ),
            army_count=territory.army_count,
        )
        for territory in state.territories
    )

    def remap_player_id(player_id: str | None) -> str | None:
        return player_mapping[player_id] if player_id is not None else None

    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "players": tuple(players),
            "territories": territories,
            "active_player_id": remap_player_id(state.active_player_id),
            "turn_player_id": remap_player_id(state.turn_player_id),
            "decision_player_id": remap_player_id(state.decision_player_id),
            "initial_placement_order": tuple(
                player_mapping[player_id]
                for player_id in state.initial_placement_order
            ),
        }
    )
    return FullGameState.model_validate(state_data)


def _topology_state(phase_id: Literal["attack", "movement"]) -> FullGameState:
    movement_phase = phase_id == "movement"
    owners = {
        territory_id: PLAYER_IDS[1] for territory_id in TERRITORY_IDS
    }
    owners["T01"] = PLAYER_IDS[0]
    if movement_phase:
        owners["T02"] = PLAYER_IDS[0]
    army_counts = {"T01": 3}

    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=owners[territory_id],
            army_count=army_counts.get(territory_id, 1),
        )
        for territory_id in TERRITORY_IDS
    )
    players: list[FullPlayerState] = []
    for player_id, color, objective_id in zip(
        PLAYER_IDS,
        PLAYER_COLORS,
        ("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-12"),
        strict=True,
    ):
        owned_territories = tuple(
            territory
            for territory in territories
            if territory.owner_player_id == player_id
        )
        players.append(
            FullPlayerState(
                public=PublicPlayerState(
                    player_id=player_id,
                    color=color,
                    territory_count=len(owned_territories),
                    army_count=sum(
                        territory.army_count for territory in owned_territories
                    ),
                    is_eliminated=not owned_territories,
                ),
                private=PlayerPrivateState(
                    player_id=player_id,
                    objective_id=objective_id,
                    hand_card_ids=(),
                ),
            )
        )
    return FullGameState(
        contract_type="full_game_state",
        game_id=f"metamorphic-{phase_id}",
        ruleset_id=WAR1_BOARD_SPEC.board_id,
        rules_version="1.0.0",
        state_version=0,
        phase_id=phase_id,
        turn_number=2,
        active_player_id=PLAYER_IDS[0],
        turn_player_id=PLAYER_IDS[0],
        decision_player_id=PLAYER_IDS[0],
        initial_placement_order=PLAYER_IDS,
        players=tuple(players),
        territories=territories,
        deck=DeckState(
            draw_pile_card_ids=CARD_IDS,
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(111).snapshot(),
        is_finished=False,
    )


def _reindexed_board(rotation: int, reverse_edges: bool) -> BoardSpec:
    original_map = WAR1_BOARD_SPEC.map
    territory_offset = rotation % len(original_map.territories)
    territories = (
        original_map.territories[territory_offset:]
        + original_map.territories[:territory_offset]
    )
    adjacency_offset = rotation % len(original_map.adjacencies)
    adjacencies = (
        original_map.adjacencies[adjacency_offset:]
        + original_map.adjacencies[:adjacency_offset]
    )
    if reverse_edges:
        adjacencies = tuple(
            Adjacency(
                first_territory_id=adjacency.second_territory_id,
                second_territory_id=adjacency.first_territory_id,
                source_references=adjacency.source_references,
            )
            for adjacency in reversed(adjacencies)
        )
    reindexed_map = MapSpec(
        map_id=original_map.map_id,
        version=original_map.version,
        territories=territories,
        continents=tuple(reversed(original_map.continents)),
        adjacencies=adjacencies,
        source_references=original_map.source_references,
    )
    board_data = WAR1_BOARD_SPEC.model_dump(mode="python")
    board_data["map"] = reindexed_map
    return BoardSpec.model_validate(board_data)


def test_legal_actions_and_transitions_use_the_supplied_board_topology() -> None:
    original_map = WAR1_BOARD_SPEC.map
    removed_adjacency = next(
        adjacency
        for adjacency in original_map.adjacencies
        if {
            adjacency.first_territory_id,
            adjacency.second_territory_id,
        }
        == {"T01", "T02"}
    )
    alternate_adjacencies = tuple(
        adjacency
        for adjacency in original_map.adjacencies
        if adjacency != removed_adjacency
    ) + (
        Adjacency(
            first_territory_id="T01",
            second_territory_id="T20",
            source_references=removed_adjacency.source_references,
        ),
    )
    alternate_map = MapSpec.model_validate(
        {
            **original_map.model_dump(mode="python"),
            "adjacencies": alternate_adjacencies,
        }
    )
    alternate_board = WAR1_BOARD_SPEC.model_copy(update={"map": alternate_map})
    state = _topology_state("attack")

    custom_action = next(
        action
        for action in legal_actions(state, board=alternate_board)
        if isinstance(action.operation, AttackOperation)
        and action.operation.origin_territory_id == "T01"
        and action.operation.target_territory_id == "T20"
    )
    with pytest.raises(InvalidActionError, match="directly adjacent"):
        validate_action(state, custom_action)

    result = apply_action(state, custom_action, board=alternate_board)

    assert result.pending_combat is not None
    assert result.pending_combat.target_territory_id == "T20"


def test_claim_legal_actions_validate_against_the_supplied_board() -> None:
    territory_claim_state = _topology_state("attack")
    state_data = territory_claim_state.model_dump(mode="python")
    claim_territory_id = "T05"
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                None
                if territory.territory_id == claim_territory_id
                else territory.owner_player_id
            ),
            army_count=(
                0
                if territory.territory_id == claim_territory_id
                else territory.army_count
            ),
        )
        for territory in territory_claim_state.territories
    )
    updated_players: list[FullPlayerState] = []
    for player in territory_claim_state.players:
        owned = tuple(
            territory
            for territory in territories
            if territory.owner_player_id == player.public.player_id
        )
        updated_players.append(
            FullPlayerState(
                public=PublicPlayerState.model_validate(
                    {
                        **player.public.model_dump(),
                        "territory_count": len(owned),
                        "army_count": sum(item.army_count for item in owned),
                        "is_eliminated": not owned,
                    }
                ),
                private=player.private,
            )
        )
    state_data.update(
        {
            "phase_id": "territory_claim",
            "turn_number": 1,
            "players": tuple(updated_players),
            "territories": territories,
        }
    )
    state = FullGameState.model_validate(state_data)

    alternate_map = WAR1_BOARD_SPEC.map.model_copy(
        update={
            "territories": tuple(
                territory
                for territory in WAR1_BOARD_SPEC.map.territories
                if territory.territory_id != claim_territory_id
            )
        }
    )
    alternate_board = WAR1_BOARD_SPEC.model_copy(update={"map": alternate_map})

    with pytest.raises(InvalidActionError, match="absent from BoardSpec"):
        legal_actions(state, board=alternate_board)
    assert any(
        action.operation.action_type == "claim_territory"
        and action.operation.territory_id == claim_territory_id
        for action in legal_actions(state, board=WAR1_BOARD_SPEC)
    )


@pytest.mark.parametrize("edge", ADJACENT_PAIRS)
def test_canonical_adjacency_is_symmetric(
    edge: tuple[str, str],
) -> None:
    first_territory_id, second_territory_id = edge

    assert _adjacent(
        WAR1_BOARD_SPEC.map,
        first_territory_id,
        second_territory_id,
    )
    assert _adjacent(
        WAR1_BOARD_SPEC.map,
        second_territory_id,
        first_territory_id,
    )


@settings(max_examples=60, deadline=None)
@given(
    player_order=st.permutations(PLAYER_IDS),
    color_order=st.permutations(PHYSICAL_COLORS),
    target_eliminated=st.booleans(),
)
def test_player_and_color_relabeling_preserves_objective_winners(
    player_order: tuple[str, ...],
    color_order: tuple[str, ...],
    target_eliminated: bool,
) -> None:
    state = _objective_state(target_eliminated)
    player_mapping = dict(zip(PLAYER_IDS, player_order, strict=True))
    color_mapping = dict(zip(PHYSICAL_COLORS, color_order, strict=True))
    relabeled_state = _relabel_players_and_colors(
        state,
        player_mapping,
        color_mapping,
    )

    winners = objective_winners(state)
    relabeled_winners = objective_winners(relabeled_state)

    assert winners == ((PLAYER_IDS[0],) if target_eliminated else ())
    assert relabeled_winners == tuple(player_mapping[player_id] for player_id in winners)


@pytest.mark.parametrize("phase_id", ("attack", "movement"))
@settings(max_examples=40, deadline=None)
@given(
    rotation=st.integers(min_value=0, max_value=78),
    reverse_edges=st.booleans(),
)
def test_reindexing_map_data_preserves_legal_actions_and_transition(
    phase_id: Literal["attack", "movement"],
    rotation: int,
    reverse_edges: bool,
) -> None:
    state = _topology_state(phase_id)
    reindexed_board = _reindexed_board(rotation, reverse_edges)
    base_action_signatures = tuple(
        sorted(
            action.operation.model_dump_json()
            for action in legal_actions(state, board=WAR1_BOARD_SPEC)
        )
    )
    reindexed_action_signatures = tuple(
        sorted(
            action.operation.model_dump_json()
            for action in legal_actions(state, board=reindexed_board)
        )
    )
    assert reindexed_action_signatures == base_action_signatures

    operation: ActionOperation
    if phase_id == "attack":
        operation = AttackOperation(
            action_type="attack",
            origin_territory_id="T01",
            target_territory_id="T02",
            dice_count=1,
        )
    else:
        operation = MoveArmiesOperation(
            action_type="move_armies",
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=1,
        )
    action = Action(
        contract_type="action",
        action_id="metamorphic-transition",
        actor_player_id=PLAYER_IDS[0],
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=operation,
    )

    original_result = apply_action(state, action, board=WAR1_BOARD_SPEC)
    reindexed_result = apply_action(state, action, board=reindexed_board)

    assert reindexed_result == original_result


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    hidden_card_indexes=st.lists(
        st.integers(min_value=0, max_value=len(CARD_IDS) - 1),
        min_size=4,
        max_size=4,
        unique=True,
    ),
)
def test_other_players_hidden_information_does_not_change_observation(
    seed: int,
    hidden_card_indexes: list[int],
) -> None:
    setup = prepare_game(
        "metamorphic-hidden-information",
        (
            SetupPlayer(player_id=PLAYER_IDS[0], color="red"),
            SetupPlayer(player_id=PLAYER_IDS[1], color="blue"),
            SetupPlayer(player_id=PLAYER_IDS[2], color="green"),
        ),
        seed,
    )
    state = setup.state
    observed_player_id = state.decision_player_id
    assert observed_player_id is not None
    hidden_player_ids = tuple(
        player_id for player_id in PLAYER_IDS if player_id != observed_player_id
    )
    hidden_cards = tuple(CARD_IDS[index] for index in hidden_card_indexes)
    observed_hand: tuple[str, ...] = ()
    first_hidden_hand = hidden_cards[:2]
    second_hidden_hand = hidden_cards[2:]
    objective_by_player = {
        observed_player_id: "W1-OBJ-13",
        hidden_player_ids[0]: "W1-OBJ-07",
        hidden_player_ids[1]: "W1-OBJ-08",
    }
    hand_by_player = {
        observed_player_id: observed_hand,
        hidden_player_ids[0]: first_hidden_hand,
        hidden_player_ids[1]: second_hidden_hand,
    }

    base_players: list[FullPlayerState] = []
    for player in state.players:
        player_id = player.public.player_id
        base_players.append(
            FullPlayerState(
                public=player.public,
                private=PlayerPrivateState(
                    player_id=player_id,
                    objective_id=objective_by_player[player_id],
                    hand_card_ids=hand_by_player[player_id],
                ),
            )
        )

    used_cards = set(hidden_cards)
    base_deck = DeckState(
        draw_pile_card_ids=tuple(
            card_id for card_id in state.deck.draw_pile_card_ids
            if card_id not in used_cards
        ),
        discard_pile_card_ids=(),
        trade_count=state.deck.trade_count,
    )
    base_data = state.model_dump(mode="python")
    base_data.update({"players": tuple(base_players), "deck": base_deck})
    visible_state = FullGameState.model_validate(base_data)

    alternative_players = tuple(
        FullPlayerState(
            public=player.public,
            private=PlayerPrivateState(
                player_id=player.public.player_id,
                objective_id=(
                    "W1-OBJ-08"
                    if player.public.player_id == hidden_player_ids[0]
                    else (
                        "W1-OBJ-07"
                        if player.public.player_id == hidden_player_ids[1]
                        else "W1-OBJ-13"
                    )
                ),
                hand_card_ids=(
                    second_hidden_hand
                    if player.public.player_id == hidden_player_ids[0]
                    else (
                        first_hidden_hand
                        if player.public.player_id == hidden_player_ids[1]
                        else observed_hand
                    )
                ),
            ),
        )
        for player in visible_state.players
    )
    alternative_data = visible_state.model_dump(mode="python")
    alternative_data.update(
        {
            "players": alternative_players,
            "random_streams_state": RandomStreams(seed + 1).snapshot(),
        }
    )
    alternative_state = FullGameState.model_validate(alternative_data)
    visible_legal_actions = legal_actions(visible_state)
    alternative_legal_actions = legal_actions(alternative_state)

    visible_observation = observe_player(
        visible_state,
        observed_player_id,
        legal_actions=visible_legal_actions,
    )
    alternative_observation = observe_player(
        alternative_state,
        observed_player_id,
        legal_actions=alternative_legal_actions,
    )

    assert visible_state != alternative_state
    assert to_public_game_state(visible_state) == to_public_game_state(
        alternative_state
    )
    assert visible_legal_actions == alternative_legal_actions
    assert visible_observation == alternative_observation
    serialized_observation = visible_observation.model_dump_json()
    assert "W1-OBJ-07" not in serialized_observation
    assert "W1-OBJ-08" not in serialized_observation
