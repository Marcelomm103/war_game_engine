"""Reproducible initial setup for the WAR 1 classic profile."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import Field

from .board_spec import WAR1_BOARD_SPEC
from .contracts import (
    BoardSpec,
    ContractModel,
    DeckState,
    FullGameState,
    FullPlayerState,
    MovementControl,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
)
from .objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    PlayerColor,
)
from .randomness import RandomStreams
from .reinforcements import calculate_reinforcement_pools
from .versioning import RULES_VERSION

PLAYER_COUNT_RANGE = (3, 6)


class SetupPlayer(ContractModel):
    player_id: str = Field(min_length=1)
    color: PlayerColor


class InvalidSetupError(ValueError):
    """Raised when player input or canonical setup data is invalid."""


@dataclass(frozen=True)
class SetupResult:
    state: FullGameState
    distributor_player_id: str
    distributor_roll_rounds: tuple[tuple[tuple[str, int], ...], ...]
    territory_deal_order: tuple[str, ...]
    territory_recipients: tuple[str, ...]
    eligible_objective_ids: tuple[str, ...]
    objective_draw_order: tuple[str, ...]


def _choose_distributor(
    player_ids: tuple[str, ...],
    roll_die: Callable[[str], int],
) -> tuple[str, tuple[tuple[tuple[str, int], ...], ...]]:
    contenders = list(player_ids)
    rounds: list[tuple[tuple[str, int], ...]] = []
    while True:
        rolls = tuple((player_id, roll_die(player_id)) for player_id in contenders)
        if any(isinstance(face, bool) or not 1 <= face <= 6 for _, face in rolls):
            raise InvalidSetupError("Distributor die rolls must be between 1 and 6.")
        rounds.append(rolls)
        highest = max(face for _, face in rolls)
        winners = tuple(player_id for player_id, face in rolls if face == highest)
        if len(winners) == 1:
            return winners[0], tuple(rounds)
        contenders = list(winners)


def _fallback_objective_id(objective_id: str, player_color: PlayerColor) -> str | None:
    if ELIMINATION_OBJECTIVE_COLORS.get(objective_id) == player_color:
        return FALLBACK_OBJECTIVE_ID
    return None


def _validate_players(players: tuple[SetupPlayer, ...]) -> None:
    minimum, maximum = PLAYER_COUNT_RANGE
    if not minimum <= len(players) <= maximum:
        raise InvalidSetupError(
            f"WAR 1 classic requires between {minimum} and {maximum} players."
        )
    player_ids = tuple(player.player_id for player in players)
    colors = tuple(player.color for player in players)
    if any(not player_id.strip() for player_id in player_ids):
        raise InvalidSetupError("Player IDs must not be blank.")
    if len(set(player_ids)) != len(player_ids):
        raise InvalidSetupError("Player IDs must be unique.")
    if len(set(colors)) != len(colors):
        raise InvalidSetupError("Player colors must be unique.")


def prepare_game(
    game_id: str,
    players: Sequence[SetupPlayer],
    seed: int,
    *,
    board: BoardSpec = WAR1_BOARD_SPEC,
) -> SetupResult:
    """Deal objectives and territories, then expose initial placement decisions."""
    if not game_id.strip():
        raise InvalidSetupError("Game ID must not be blank.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InvalidSetupError("Setup seed must be a non-negative integer.")
    player_specs = tuple(players)
    _validate_players(player_specs)

    if len(board.map.territories) != 42 or len(board.objectives) != 14:
        raise InvalidSetupError(
            "WAR 1 classic setup requires 42 territories and 14 objectives."
        )
    objective_ids = {objective.objective_id for objective in board.objectives}
    required_objective_ids = set(ELIMINATION_OBJECTIVE_COLORS).union(
        {FALLBACK_OBJECTIVE_ID}
    )
    missing_objectives = sorted(required_objective_ids - objective_ids)
    if missing_objectives:
        raise InvalidSetupError(
            f"BoardSpec is missing setup objectives: {', '.join(missing_objectives)}."
        )

    streams = RandomStreams(seed)
    player_ids = tuple(player.player_id for player in player_specs)
    active_colors = {player.color for player in player_specs}
    eligible_objectives = tuple(
        objective
        for objective in board.objectives
        if objective.objective_id not in ELIMINATION_OBJECTIVE_COLORS
        or ELIMINATION_OBJECTIVE_COLORS[objective.objective_id] in active_colors
    )
    expected_objective_count = 8 + len(player_specs)
    if len(eligible_objectives) != expected_objective_count:
        raise InvalidSetupError(
            "Objective filtering produced an unexpected number of eligible cards: "
            f"{len(eligible_objectives)} instead of {expected_objective_count}."
        )
    if len(eligible_objectives) < len(player_specs):
        raise InvalidSetupError(
            "Not enough eligible objectives for all participating players."
        )

    objective_draw = streams.sample(
        "objectives", eligible_objectives, len(player_specs)
    )
    objective_by_player = {
        player.player_id: objective.objective_id
        for player, objective in zip(player_specs, objective_draw, strict=True)
    }

    distributor_player_id, distributor_roll_rounds = _choose_distributor(
        player_ids,
        lambda _player_id: streams.die("preparation"),
    )
    player_index = {
        player.player_id: index for index, player in enumerate(player_specs)
    }
    distributor_index = player_index[distributor_player_id]
    distribution_start_index = (distributor_index + 1) % len(player_specs)
    distribution_order = tuple(
        player_specs[(distribution_start_index + index) % len(player_specs)].player_id
        for index in range(len(player_specs))
    )

    wildcard_card_ids = set(board.wildcard_card_ids)
    territory_cards = tuple(
        card_id
        for card_id in board.territory_deck.card_ids
        if card_id not in wildcard_card_ids
    )
    territory_ids = {territory.territory_id for territory in board.map.territories}
    if len(territory_cards) != 42 or set(territory_cards) != territory_ids:
        raise InvalidSetupError(
            "Territory deck must contain one card for each canonical territory."
        )
    territory_deal_order = streams.shuffle("preparation", territory_cards)

    territory_owner: dict[str, str] = {}
    territory_recipients: list[str] = []
    for index, territory_id in enumerate(territory_deal_order):
        recipient = distribution_order[index % len(distribution_order)]
        territory_owner[territory_id] = recipient
        territory_recipients.append(recipient)

    last_recipient = territory_recipients[-1]
    first_placement_index = (player_index[last_recipient] + 1) % len(player_specs)
    initial_placement_order = tuple(
        player_specs[(first_placement_index + index) % len(player_specs)].player_id
        for index in range(len(player_specs))
    )

    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=territory_owner[territory.territory_id],
            army_count=1,
        )
        for territory in board.map.territories
    )
    territories_by_player = {
        player.player_id: tuple(
            territory
            for territory in territories
            if territory.owner_player_id == player.player_id
        )
        for player in player_specs
    }
    reinforcement_pools: list[ReinforcementPool] = []
    for player in player_specs:
        reinforcement_pools.extend(
            calculate_reinforcement_pools(
                player.player_id,
                territories,
                board.map,
                pool_scope="initial",
            )
        )

    full_players = tuple(
        FullPlayerState(
            public=PublicPlayerState(
                player_id=player.player_id,
                color=player.color,
                territory_count=len(territories_by_player[player.player_id]),
                army_count=sum(
                    territory.army_count
                    for territory in territories_by_player[player.player_id]
                ),
                is_eliminated=False,
            ),
            private=PlayerPrivateState(
                player_id=player.player_id,
                objective_id=objective_by_player[player.player_id],
                objective_fallback_id=_fallback_objective_id(
                    objective_by_player[player.player_id],
                    player.color,
                ),
                hand_card_ids=(),
            ),
        )
        for player in player_specs
    )

    first_player_id = initial_placement_order[0]
    shuffled_game_deck = streams.shuffle("cards", board.territory_deck.card_ids)
    state = FullGameState(
        contract_type="full_game_state",
        game_id=game_id,
        ruleset_id=board.board_id,
        rules_version=RULES_VERSION,
        state_version=0,
        phase_id="initial_placement",
        turn_number=1,
        active_player_id=first_player_id,
        turn_player_id=first_player_id,
        decision_player_id=first_player_id,
        initial_placement_order=initial_placement_order,
        players=full_players,
        territories=territories,
        reinforcement_pools=tuple(reinforcement_pools),
        movement_control=MovementControl(),
        deck=DeckState(
            draw_pile_card_ids=shuffled_game_deck,
            discard_pile_card_ids=(),
            trade_count=0,
        ),
        engine_random_state=None,
        random_streams_state=streams.snapshot(),
        is_finished=False,
    )
    return SetupResult(
        state=state,
        distributor_player_id=distributor_player_id,
        distributor_roll_rounds=distributor_roll_rounds,
        territory_deal_order=territory_deal_order,
        territory_recipients=tuple(territory_recipients),
        eligible_objective_ids=tuple(
            objective.objective_id for objective in eligible_objectives
        ),
        objective_draw_order=tuple(
            objective.objective_id for objective in objective_draw
        ),
    )


__all__ = [
    "ELIMINATION_OBJECTIVE_COLORS",
    "FALLBACK_OBJECTIVE_ID",
    "InvalidSetupError",
    "PlayerColor",
    "SetupPlayer",
    "SetupResult",
    "prepare_game",
]
