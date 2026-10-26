import pytest

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import TerritoryState
from war_game_engine.reinforcements import calculate_reinforcement_pools


def test_inactive_players_cannot_receive_reinforcement_pools() -> None:
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=None,
            army_count=0,
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    )

    with pytest.raises(ValueError, match="inactive player"):
        calculate_reinforcement_pools(
            "eliminated",
            territories,
            WAR1_BOARD_SPEC.map,
            pool_scope="test",
        )


def test_zero_continent_bonus_does_not_create_a_pool() -> None:
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id="player-1",
            army_count=1,
        )
        for territory in WAR1_BOARD_SPEC.map.territories
    )
    zero_bonus_map = WAR1_BOARD_SPEC.map.model_copy(
        update={
            "continents": tuple(
                continent.model_copy(update={"reinforcement_bonus": 0})
                if continent.continent_id == "oceania"
                else continent
                for continent in WAR1_BOARD_SPEC.map.continents
            )
        }
    )

    pools = calculate_reinforcement_pools(
        "player-1",
        territories,
        zero_bonus_map,
        pool_scope="test",
    )

    assert all(
        pool.allowed_territory_ids
        != ("T01", "T02", "T03", "T04")
        for pool in pools
        if pool.origin == "continent_bonus"
    )
