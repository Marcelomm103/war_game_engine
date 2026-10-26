from collections import Counter
from collections.abc import Mapping
from typing import TypeAlias

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    ActionOperation,
    DeckState,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    FullGameState,
    FullPlayerState,
    GamePhaseId,
    MoveArmiesOperation,
    MovementControl,
    MovementTransfer,
    PendingObjectiveCheck,
    PendingOccupation,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
    TradeCardsOperation,
)
from war_game_engine.engine import (
    _transfer_eliminated_cards,
    apply_action,
    legal_actions,
    resolve_pending_objective_check,
    validate_action,
)
from war_game_engine.objectives import objective_is_satisfied, objective_winners
from war_game_engine.randomness import RandomStreams
from war_game_engine.versioning import ACTION_SCHEMA_VERSION

PLAYER_IDS = ("player-1", "player-2", "player-3")
PLAYER_COLORS = ("white", "red", "black")
OBJECTIVE_IDS = ("W1-OBJ-13", "W1-OBJ-14", "W1-OBJ-12")
TERRITORY_IDS = tuple(
    territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
)
CARD_IDS = WAR1_BOARD_SPEC.territory_deck.card_ids
ADJACENT_PAIRS = tuple(
    (adjacency.first_territory_id, adjacency.second_territory_id)
    for adjacency in WAR1_BOARD_SPEC.map.adjacencies
)
ALL_GAME_PHASES: tuple[GamePhaseId, ...] = (
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
)
PHASE_ACTION_TYPES: dict[GamePhaseId, frozenset[str]] = {
    "territory_claim": frozenset({"claim_territory"}),
    "initial_placement": frozenset({"place_armies"}),
    "reinforcement": frozenset({"trade_cards", "end_phase"}),
    "reinforcement_placement": frozenset({"place_armies"}),
    "attack": frozenset({"attack", "end_phase"}),
    "occupation": frozenset({"occupy_territory"}),
    "objective_check": frozenset(),
    "movement": frozenset({"move_armies", "end_phase"}),
    "card_purchase": frozenset({"draw_territory_card"}),
    "card_discard": frozenset({"discard_territory_card"}),
    "turn_end": frozenset({"end_turn"}),
    "finished": frozenset(),
}
TradeSet: TypeAlias = tuple[str, str, str]
VALID_TRADE_SETS: tuple[TradeSet, ...] = (
    ("T01", "T02", "T08"),
    ("T01", "T05", "T09"),
    ("T01", "T02", "JOKER-01"),
    ("JOKER-01", "JOKER-02", "T01"),
)


def _make_action(
    state: FullGameState,
    operation: ActionOperation,
    *,
    actor_player_id: str = PLAYER_IDS[0],
) -> Action:
    return Action(
        contract_type="action",
        action_id=f"property-{state.state_version}",
        actor_player_id=actor_player_id,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=operation,
    )


def _build_state(
    phase_id: GamePhaseId,
    owner_by_territory: Mapping[str, str | None],
    *,
    army_counts: Mapping[str, int] | None = None,
    turn_number: int = 2,
    state_version: int = 0,
    active_player_id: str | None = PLAYER_IDS[0],
    decision_player_id: str | None = PLAYER_IDS[0],
    pools: tuple[ReinforcementPool, ...] = (),
    objective_ids: tuple[str, str, str] = OBJECTIVE_IDS,
    hands_by_player: Mapping[str, tuple[str, ...]] | None = None,
    draw_pile_card_ids: tuple[str, ...] | None = None,
    discard_pile_card_ids: tuple[str, ...] = (),
    hand_limit_discard_card_ids: tuple[str, ...] = (),
    trade_count: int = 0,
    has_conquered_territory_this_turn: bool = False,
    pending_occupation: PendingOccupation | None = None,
    pending_objective_check: PendingObjectiveCheck | None = None,
    winner_player_id: str | None = None,
    is_finished: bool = False,
    eliminated_by_player_id: Mapping[str, str] | None = None,
) -> FullGameState:
    armies = {} if army_counts is None else army_counts
    hands = {} if hands_by_player is None else hands_by_player
    eliminators = (
        {} if eliminated_by_player_id is None else eliminated_by_player_id
    )
    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=owner_by_territory[territory_id],
            army_count=armies.get(
                territory_id,
                1 if owner_by_territory[territory_id] is not None else 0,
            ),
        )
        for territory_id in TERRITORY_IDS
    )
    players: list[FullPlayerState] = []
    for player_id, color, objective_id in zip(
        PLAYER_IDS,
        PLAYER_COLORS,
        objective_ids,
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
                    eliminated_by_player_id=eliminators.get(player_id),
                ),
                private=PlayerPrivateState(
                    player_id=player_id,
                    objective_id=objective_id,
                    hand_card_ids=hands.get(player_id, ()),
                ),
            )
        )

    used_card_ids = {
        card_id
        for player_hand in hands.values()
        for card_id in player_hand
    } | set(discard_pile_card_ids) | set(hand_limit_discard_card_ids)
    draw_cards = (
        tuple(
            card_id for card_id in CARD_IDS if card_id not in used_card_ids
        )
        if draw_pile_card_ids is None
        else draw_pile_card_ids
    )
    return FullGameState(
        contract_type="full_game_state",
        game_id="property-test",
        ruleset_id=WAR1_BOARD_SPEC.board_id,
        rules_version="1.0.0",
        state_version=state_version,
        phase_id=phase_id,
        turn_number=turn_number,
        has_conquered_territory_this_turn=has_conquered_territory_this_turn,
        active_player_id=active_player_id,
        turn_player_id=active_player_id,
        decision_player_id=decision_player_id,
        initial_placement_order=PLAYER_IDS,
        players=tuple(players),
        territories=territories,
        reinforcement_pools=pools,
        movement_control=MovementControl(),
        pending_occupation=pending_occupation,
        pending_objective_check=pending_objective_check,
        winner_player_id=winner_player_id,
        deck=DeckState(
            draw_pile_card_ids=draw_cards,
            discard_pile_card_ids=discard_pile_card_ids,
            hand_limit_discard_card_ids=hand_limit_discard_card_ids,
            trade_count=trade_count,
        ),
        engine_random_state=None,
        random_streams_state=RandomStreams(2026).snapshot(),
        is_finished=is_finished,
    )


def _assert_game_invariants(state: FullGameState) -> None:
    players_by_id = {
        player.public.player_id: player.public for player in state.players
    }
    territory_counts = dict.fromkeys(players_by_id, 0)
    army_counts = dict.fromkeys(players_by_id, 0)
    for territory in state.territories:
        assert territory.army_count >= 0
        if territory.owner_player_id is None:
            assert territory.army_count == 0
            continue
        owner = players_by_id[territory.owner_player_id]
        assert not owner.is_eliminated
        assert territory.army_count >= 1
        territory_counts[owner.player_id] += 1
        army_counts[owner.player_id] += territory.army_count

    for player in state.players:
        public = player.public
        assert public.territory_count == territory_counts[public.player_id]
        assert public.army_count == army_counts[public.player_id]
        assert public.is_eliminated == (public.territory_count == 0)
        if public.is_eliminated:
            assert not any(
                pool.player_id == public.player_id
                for pool in state.reinforcement_pools
            )
            assert public.player_id not in {
                state.active_player_id,
                state.turn_player_id,
                state.decision_player_id,
            }

    cards = [
        *state.deck.draw_pile_card_ids,
        *state.deck.discard_pile_card_ids,
        *state.deck.hand_limit_discard_card_ids,
        *(
            card_id
            for player in state.players
            for card_id in player.private.hand_card_ids
        ),
    ]
    assert len(cards) == len(set(cards))
    assert set(cards) == set(CARD_IDS)
    for player in state.players:
        is_current_discard = (
            state.phase_id == "card_discard"
            and player.public.player_id == state.decision_player_id
        )
        assert len(player.private.hand_card_ids) <= (6 if is_current_discard else 5)

    assert state.is_finished == (state.phase_id == "finished")
    if state.is_finished:
        assert state.winner_player_id is not None
        assert state.active_player_id is None
        assert state.turn_player_id is None
        assert state.decision_player_id is None
    else:
        assert state.winner_player_id is None
        assert state.active_player_id is not None
        assert state.turn_player_id == state.active_player_id
        assert state.decision_player_id is not None
        assert not players_by_id[state.decision_player_id].is_eliminated
    for pool in state.reinforcement_pools:
        assert not players_by_id[pool.player_id].is_eliminated


def _available_armies(state: FullGameState) -> int:
    return sum(territory.army_count for territory in state.territories) + sum(
        pool.remaining_armies for pool in state.reinforcement_pools
    )


def _card_inventory(state: FullGameState) -> Counter[str]:
    return Counter(
        (
            *state.deck.draw_pile_card_ids,
            *state.deck.discard_pile_card_ids,
            *state.deck.hand_limit_discard_card_ids,
            *(
                card_id
                for player in state.players
                for card_id in player.private.hand_card_ids
            ),
        )
    )


def _phase_state(
    phase_id: GamePhaseId,
    *,
    state_version: int = 0,
) -> FullGameState:
    owners: dict[str, str | None] = {
        territory_id: PLAYER_IDS[1] for territory_id in TERRITORY_IDS
    }
    owners["T01"] = PLAYER_IDS[0]
    armies: dict[str, int] = {"T01": 3}
    pools: tuple[ReinforcementPool, ...] = ()
    hands: dict[str, tuple[str, ...]] = {}
    turn_number = 2
    active_player_id: str | None = PLAYER_IDS[0]
    decision_player_id: str | None = PLAYER_IDS[0]
    has_conquered = False
    pending_occupation: PendingOccupation | None = None
    pending_objective_check: PendingObjectiveCheck | None = None
    winner_player_id: str | None = None
    is_finished = False

    if phase_id == "territory_claim":
        turn_number = 1
        owners["T03"] = None
    elif phase_id == "initial_placement":
        turn_number = 1
        owners["T03"] = PLAYER_IDS[2]
        pools = tuple(
            ReinforcementPool(
                pool_id=f"{player_id}:initial",
                player_id=player_id,
                origin="territory_count",
                remaining_armies=1,
            )
            for player_id in PLAYER_IDS
        )
    elif phase_id == "reinforcement":
        pools = (
            ReinforcementPool(
                pool_id="player-1:reinforcement",
                player_id=PLAYER_IDS[0],
                origin="territory_count",
                remaining_armies=3,
            ),
        )
        hands = {PLAYER_IDS[0]: ("T01", "T02", "T08")}
    elif phase_id == "reinforcement_placement":
        pools = (
            ReinforcementPool(
                pool_id="player-1:placement",
                player_id=PLAYER_IDS[0],
                origin="territory_count",
                remaining_armies=3,
            ),
        )
    elif phase_id == "attack":
        owners["T02"] = PLAYER_IDS[1]
    elif phase_id == "occupation":
        owners["T02"] = None
        pending_occupation = PendingOccupation(
            contract_type="pending_occupation",
            attacker_player_id=PLAYER_IDS[0],
            defender_player_id=PLAYER_IDS[1],
            origin_territory_id="T01",
            target_territory_id="T02",
            maximum_army_count=2,
        )
    elif phase_id == "objective_check":
        pending_objective_check = PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=PLAYER_IDS[0],
            territory_id="T02",
            resume_phase_id="attack",
            resume_decision_player_id=PLAYER_IDS[0],
        )
    elif phase_id == "movement":
        owners["T02"] = PLAYER_IDS[0]
        armies["T02"] = 1
    elif phase_id == "card_purchase":
        has_conquered = True
    elif phase_id == "card_discard":
        has_conquered = True
        hands = {PLAYER_IDS[0]: CARD_IDS[:6]}
    elif phase_id == "finished":
        for index, territory_id in enumerate(TERRITORY_IDS):
            owners[territory_id] = (
                PLAYER_IDS[0] if index < 24 else PLAYER_IDS[1]
            )
        active_player_id = None
        decision_player_id = None
        winner_player_id = PLAYER_IDS[0]
        is_finished = True

    return _build_state(
        phase_id,
        owners,
        army_counts=armies,
        turn_number=turn_number,
        state_version=state_version,
        active_player_id=active_player_id,
        decision_player_id=decision_player_id,
        pools=pools,
        hands_by_player=hands,
        has_conquered_territory_this_turn=has_conquered,
        pending_occupation=pending_occupation,
        pending_objective_check=pending_objective_check,
        winner_player_id=winner_player_id,
        is_finished=is_finished,
    )


@pytest.mark.parametrize("phase_id", ALL_GAME_PHASES)
@settings(max_examples=8, deadline=None)
@given(state_version=st.integers(min_value=0, max_value=2**63))
def test_legal_actions_match_the_phase_and_preserve_state_invariants(
    phase_id: GamePhaseId,
    state_version: int,
) -> None:
    state = _phase_state(phase_id, state_version=state_version)
    original_state = state.model_dump_json()
    actions = legal_actions(state)
    expected_action_types = PHASE_ACTION_TYPES[phase_id]

    if not expected_action_types:
        assert actions == ()
        return

    assert actions
    for action in actions:
        assert action.actor_player_id == state.decision_player_id
        assert action.expected_state_version == state.state_version
        assert action.operation.action_type in expected_action_types
        validate_action(state, action)
        result = apply_action(state, action)
        _assert_game_invariants(result)

    assert state.model_dump_json() == original_state


@settings(max_examples=60, deadline=None)
@given(
    pool_armies=st.integers(min_value=1, max_value=100),
    target_index=st.integers(min_value=0, max_value=100),
    starting_armies=st.integers(min_value=1, max_value=50),
)
def test_initial_placement_conserves_troops_and_respects_the_pool(
    pool_armies: int,
    target_index: int,
    starting_armies: int,
) -> None:
    owners = {
        territory_id: PLAYER_IDS[index % len(PLAYER_IDS)]
        for index, territory_id in enumerate(TERRITORY_IDS)
    }
    armies = dict.fromkeys(TERRITORY_IDS, starting_armies)
    pools = (
        ReinforcementPool(
            pool_id="player-1:initial",
            player_id=PLAYER_IDS[0],
            origin="territory_count",
            remaining_armies=pool_armies,
        ),
        ReinforcementPool(
            pool_id="player-2:initial",
            player_id=PLAYER_IDS[1],
            origin="territory_count",
            remaining_armies=1,
        ),
        ReinforcementPool(
            pool_id="player-3:initial",
            player_id=PLAYER_IDS[2],
            origin="territory_count",
            remaining_armies=1,
        ),
    )
    state = _build_state(
        "initial_placement",
        owners,
        army_counts=armies,
        turn_number=1,
        pools=pools,
    )
    player_territory_ids = tuple(
        territory.territory_id
        for territory in state.territories
        if territory.owner_player_id == PLAYER_IDS[0]
    )
    target_territory_id = player_territory_ids[
        target_index % len(player_territory_ids)
    ]
    actions = legal_actions(state)
    action = next(
        candidate
        for candidate in actions
        if isinstance(candidate.operation, PlaceArmiesOperation)
        and candidate.operation.placements[0].territory_id == target_territory_id
    )

    result = apply_action(state, action)

    _assert_game_invariants(result)
    assert _available_armies(result) == _available_armies(state)
    assert result.state_version == state.state_version + 1
    assert tuple(
        (
            territory.territory_id,
            territory.owner_player_id,
            territory.army_count,
        )
        for territory in result.territories
    ) == tuple(
        (
            territory.territory_id,
            territory.owner_player_id,
            territory.army_count
            + (territory.territory_id == target_territory_id),
        )
        for territory in state.territories
    )
    remaining_pools = {
        pool.pool_id: pool.remaining_armies for pool in result.reinforcement_pools
    }
    if pool_armies == 1:
        assert "player-1:initial" not in remaining_pools
        assert result.decision_player_id == PLAYER_IDS[1]
    else:
        assert remaining_pools["player-1:initial"] == pool_armies - 1
        assert result.decision_player_id == PLAYER_IDS[0]


@settings(max_examples=60, deadline=None)
@given(
    edge=st.sampled_from(ADJACENT_PAIRS),
    movable_armies=st.integers(min_value=1, max_value=100),
    move_selector=st.integers(min_value=0, max_value=100),
    target_armies=st.integers(min_value=1, max_value=100),
)
def test_direct_movement_conserves_troops_and_preserves_ownership(
    edge: tuple[str, str],
    movable_armies: int,
    move_selector: int,
    target_armies: int,
) -> None:
    origin_territory_id, target_territory_id = edge
    move_army_count = 1 + (move_selector % movable_armies)
    owners = {
        territory_id: PLAYER_IDS[1] for territory_id in TERRITORY_IDS
    }
    owners[origin_territory_id] = PLAYER_IDS[0]
    owners[target_territory_id] = PLAYER_IDS[0]
    armies = {
        origin_territory_id: movable_armies + 1,
        target_territory_id: target_armies,
    }
    state = _build_state(
        "movement",
        owners,
        army_counts=armies,
    )
    action = _make_action(
        state,
        MoveArmiesOperation(
            action_type="move_armies",
            origin_territory_id=origin_territory_id,
            target_territory_id=target_territory_id,
            army_count=move_army_count,
        ),
    )

    result = apply_action(state, action)

    _assert_game_invariants(result)
    assert _available_armies(result) == _available_armies(state)
    assert result.phase_id == "movement"
    assert tuple(
        territory.owner_player_id for territory in result.territories
    ) == tuple(territory.owner_player_id for territory in state.territories)
    assert tuple(
        territory.army_count for territory in result.territories
    ) == tuple(
        territory.army_count
        + (
            move_army_count
            if territory.territory_id == target_territory_id
            else (
                -move_army_count
                if territory.territory_id == origin_territory_id
                else 0
            )
        )
        for territory in state.territories
    )
    assert result.movement_control.moved_transfers == (
        MovementTransfer(
            origin_territory_id=origin_territory_id,
            target_territory_id=target_territory_id,
            army_count=move_army_count,
        ),
    )


@settings(max_examples=60, deadline=None)
@given(
    trade_cards=st.sampled_from(VALID_TRADE_SETS),
    owned_mask=st.integers(min_value=0, max_value=7),
    extra_card_count=st.integers(min_value=0, max_value=2),
    trade_count=st.integers(min_value=0, max_value=50),
)
def test_card_trades_preserve_card_inventory_and_account_for_reinforcements(
    trade_cards: TradeSet,
    owned_mask: int,
    extra_card_count: int,
    trade_count: int,
) -> None:
    territory_cards = set(TERRITORY_IDS)
    wildcard_cards = set(WAR1_BOARD_SPEC.wildcard_card_ids)
    owned_territory_cards = {
        card_id
        for index, card_id in enumerate(trade_cards)
        if card_id in territory_cards and owned_mask & (1 << index)
    }
    anchor_territory_id = next(
        territory_id for territory_id in TERRITORY_IDS if territory_id not in trade_cards
    )
    player_one_territory_ids = owned_territory_cards | {anchor_territory_id}
    owners = {
        territory_id: (
            PLAYER_IDS[0]
            if territory_id in player_one_territory_ids
            else PLAYER_IDS[1]
        )
        for territory_id in TERRITORY_IDS
    }
    available_cards = tuple(
        card_id for card_id in CARD_IDS if card_id not in trade_cards
    )
    extra_cards = available_cards[:extra_card_count]
    hands = {PLAYER_IDS[0]: (*trade_cards, *extra_cards)}
    state = _build_state(
        "reinforcement",
        owners,
        turn_number=2,
        pools=(
            ReinforcementPool(
                pool_id="player-1:reinforcement",
                player_id=PLAYER_IDS[0],
                origin="territory_count",
                remaining_armies=5,
            ),
        ),
        hands_by_player=hands,
        trade_count=trade_count,
    )
    original_inventory = _card_inventory(state)
    original_territory_armies = tuple(
        territory.army_count for territory in state.territories
    )
    original_pool_armies = sum(
        pool.remaining_armies for pool in state.reinforcement_pools
    )
    action = _make_action(
        state,
        TradeCardsOperation(action_type="trade_cards", card_ids=trade_cards),
    )

    result = apply_action(state, action)

    _assert_game_invariants(result)
    next_trade_number = trade_count + 1
    trade_values = (4, 6, 8, 10, 12, 15, 20)
    base_reinforcements = (
        trade_values[next_trade_number - 1]
        if next_trade_number <= len(trade_values)
        else 20 + 5 * (next_trade_number - 7)
    )
    territory_bonus_count = sum(
        card_id in territory_cards
        and card_id not in wildcard_cards
        and card_id in player_one_territory_ids
        for card_id in trade_cards
    )
    result_pool_armies = sum(
        pool.remaining_armies for pool in result.reinforcement_pools
    )
    assert (
        result_pool_armies - original_pool_armies
        == base_reinforcements + 2 * territory_bonus_count
    )
    assert tuple(
        territory.army_count for territory in result.territories
    ) == original_territory_armies
    assert _card_inventory(result) == original_inventory
    assert result.players[0].private.hand_card_ids == extra_cards
    assert result.deck.discard_pile_card_ids == trade_cards
    assert result.deck.trade_count == trade_count + 1


@settings(max_examples=60, deadline=None)
@given(
    top_card_index=st.integers(min_value=0, max_value=len(CARD_IDS) - 1),
    starting_hand_size=st.integers(min_value=0, max_value=4),
)
def test_card_draw_preserves_inventory_and_obeys_the_hand_limit(
    top_card_index: int,
    starting_hand_size: int,
) -> None:
    top_card = CARD_IDS[top_card_index]
    available_cards = tuple(card_id for card_id in CARD_IDS if card_id != top_card)
    hand = available_cards[:starting_hand_size]
    draw_pile = (
        top_card,
        *(
            card_id
            for card_id in available_cards
            if card_id not in hand
        ),
    )
    owners = {territory_id: PLAYER_IDS[1] for territory_id in TERRITORY_IDS}
    owners["T01"] = PLAYER_IDS[0]
    state = _build_state(
        "card_purchase",
        owners,
        has_conquered_territory_this_turn=True,
        hands_by_player={PLAYER_IDS[0]: hand},
        draw_pile_card_ids=draw_pile,
    )
    original_inventory = _card_inventory(state)
    original_armies = tuple(
        territory.army_count for territory in state.territories
    )
    action = _make_action(
        state,
        DrawTerritoryCardOperation(action_type="draw_territory_card"),
    )

    result = apply_action(state, action)

    _assert_game_invariants(result)
    assert result.players[0].private.hand_card_ids == (*hand, top_card)
    assert result.deck.draw_pile_card_ids == draw_pile[1:]
    assert result.phase_id == "turn_end"
    assert _card_inventory(result) == original_inventory
    assert tuple(territory.army_count for territory in result.territories) == (
        original_armies
    )


@settings(max_examples=60, deadline=None)
@given(discard_index=st.integers(min_value=0, max_value=100))
def test_hand_limit_discard_preserves_card_inventory(discard_index: int) -> None:
    hand = CARD_IDS[:6]
    owners = {territory_id: PLAYER_IDS[1] for territory_id in TERRITORY_IDS}
    owners["T01"] = PLAYER_IDS[0]
    state = _build_state(
        "card_discard",
        owners,
        has_conquered_territory_this_turn=True,
        hands_by_player={PLAYER_IDS[0]: hand},
    )
    original_inventory = _card_inventory(state)
    action = legal_actions(state)[discard_index % 6]
    assert isinstance(action.operation, DiscardTerritoryCardOperation)

    result = apply_action(state, action)

    _assert_game_invariants(result)
    discarded_card = action.operation.card_id
    assert result.players[0].private.hand_card_ids == tuple(
        card_id for card_id in hand if card_id != discarded_card
    )
    assert result.deck.hand_limit_discard_card_ids == (discarded_card,)
    assert result.phase_id == "turn_end"
    assert _card_inventory(result) == original_inventory


@settings(max_examples=60, deadline=None)
@given(
    recipient_hand_size=st.integers(min_value=0, max_value=5),
    eliminated_hand_size=st.integers(min_value=0, max_value=5),
)
def test_elimination_card_transfer_preserves_every_card(
    recipient_hand_size: int,
    eliminated_hand_size: int,
) -> None:
    owners = {
        territory_id: PLAYER_IDS[0] for territory_id in TERRITORY_IDS
    }
    recipient_hand = CARD_IDS[:recipient_hand_size]
    eliminated_hand = CARD_IDS[
        recipient_hand_size : recipient_hand_size + eliminated_hand_size
    ]

    state = _build_state(
        "attack",
        owners,
        hands_by_player={
            PLAYER_IDS[0]: recipient_hand,
            PLAYER_IDS[1]: eliminated_hand,
        },
    )

    players, deck, random_streams_state = _transfer_eliminated_cards(
        state,
        PLAYER_IDS[0],
        PLAYER_IDS[1],
    )
    result = FullGameState.model_validate(
        {
            **state.model_dump(mode="python"),
            "players": players,
            "deck": deck,
            "random_streams_state": random_streams_state,
        }
    )
    _assert_game_invariants(result)

    combined_hand_size = recipient_hand_size + eliminated_hand_size
    excess_card_count = max(0, combined_hand_size - 5)
    assert len(players[0].private.hand_card_ids) == min(5, combined_hand_size)
    assert players[1].private.hand_card_ids == ()
    assert len(deck.hand_limit_discard_card_ids) == excess_card_count
    assert _card_inventory(result) == _card_inventory(state)
    if excess_card_count:
        assert random_streams_state != state.random_streams_state
    else:
        assert random_streams_state == state.random_streams_state


@settings(max_examples=40, deadline=None)
@given(
    player_two_active=st.booleans(),
    player_three_active=st.booleans(),
)
def test_turn_rotation_never_selects_an_eliminated_player(
    player_two_active: bool,
    player_three_active: bool,
) -> None:
    owners = {
        territory_id: PLAYER_IDS[0] for territory_id in TERRITORY_IDS
    }
    if player_two_active:
        owners["T02"] = PLAYER_IDS[1]
    if player_three_active:
        owners["T03"] = PLAYER_IDS[2]
    state = _build_state(
        "turn_end",
        owners,
        turn_number=2,
    )

    result = apply_action(state, legal_actions(state)[0])

    expected_player_id = (
        PLAYER_IDS[1]
        if player_two_active
        else (PLAYER_IDS[2] if player_three_active else PLAYER_IDS[0])
    )
    expected_turn_number = 3 if expected_player_id == PLAYER_IDS[0] else 2
    _assert_game_invariants(result)
    assert result.decision_player_id == expected_player_id
    assert result.turn_number == expected_turn_number
    assert {
        pool.player_id for pool in result.reinforcement_pools
    } == {expected_player_id}
    for player in result.players:
        if player.public.is_eliminated:
            assert player.public.player_id not in {
                pool.player_id for pool in result.reinforcement_pools
            }


@settings(max_examples=60, deadline=None)
@given(
    owned_count=st.integers(min_value=0, max_value=len(TERRITORY_IDS)),
    fortified_selector=st.integers(min_value=0, max_value=len(TERRITORY_IDS)),
)
def test_victory_occurs_only_when_the_objective_condition_is_satisfied(
    owned_count: int,
    fortified_selector: int,
) -> None:
    territory_ids = TERRITORY_IDS
    player_one_territory_ids = set(territory_ids[:owned_count])
    owner_by_territory: dict[str, str | None] = {
        territory_id: PLAYER_IDS[0]
        for territory_id in player_one_territory_ids
    }
    remaining_by_continent: dict[str, list[str]] = {}
    for territory in WAR1_BOARD_SPEC.map.territories:
        if territory.territory_id not in player_one_territory_ids:
            remaining_by_continent.setdefault(territory.continent_id, []).append(
                territory.territory_id
            )
    for remaining_territories in remaining_by_continent.values():
        for index, territory_id in enumerate(remaining_territories):
            owner_by_territory[territory_id] = (
                PLAYER_IDS[1] if index % 2 == 0 else PLAYER_IDS[2]
            )

    fortified_count = min(fortified_selector, owned_count)
    army_counts = {
        territory_id: (
            2 if index < fortified_count else 1
        )
        for index, territory_id in enumerate(territory_ids[:owned_count])
    }
    active_player_id = (
        PLAYER_IDS[0] if owned_count else PLAYER_IDS[1]
    )
    state = _build_state(
        "objective_check",
        owner_by_territory,
        army_counts=army_counts,
        active_player_id=active_player_id,
        decision_player_id=active_player_id,
        objective_ids=("W1-OBJ-14", "W1-OBJ-13", "W1-OBJ-12"),
        pending_objective_check=PendingObjectiveCheck(
            contract_type="pending_objective_check",
            actor_player_id=active_player_id,
            territory_id=territory_ids[0],
            resume_phase_id="attack",
            resume_decision_player_id=active_player_id,
        ),
    )

    result = resolve_pending_objective_check(state)

    qualifies = owned_count >= 18 and fortified_count >= 18
    _assert_game_invariants(result)
    assert result.is_finished is qualifies
    if qualifies:
        assert result.winner_player_id == PLAYER_IDS[0]
        assert objective_is_satisfied(result, result.players[0])
    else:
        assert result.winner_player_id is None
        assert objective_winners(result) == ()
