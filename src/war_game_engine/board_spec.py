"""Canonical board and component data for the WAR 1 classic profile."""

from __future__ import annotations

from typing import TypeAlias

from .contracts import (
    Adjacency,
    BoardSpec,
    ContinentDefinition,
    DeckSpec,
    MapSpec,
    ObjectiveDefinition,
    SourceReference,
    TerritoryCardSymbol,
    TerritoryDefinition,
)

TerritoryCatalogEntry: TypeAlias = tuple[
    str, str, str, tuple[str, ...], TerritoryCardSymbol, str
]

_ASSET_SHA256 = {
    "mapa_war_classico.webp": "6877a96fa88616202eb9448e494ccb82f9c2dc2edf207daf2c33cfe6584548b8",
    "war_cartas1.png": "96e7e9eb60f894ec931b716c35d47c4a22922c25ce1ede28ac3f01ef965adde3",
    "war_cartas2.png": "b7b8aa7fed5fc40bcb5d0b3e48a707501d7ca793641a4ca40de66b6ce888ced9",
    "war_cartas3.png": "a30f23c9c74501cb884270e0043e58f7e5c344aa726c84fb8a21934816ee958a",
    "war_cartas4.png": "3eca1085759d76a5d851832ac32a9a24f7681f2c17819faade31d0445beb8b04",
    "war_cartas5.png": "ec7721e82d16eea729a67476e9b3e3923d28a5c11f0acc64b9a52d971bcba3d3",
    "war_cartas6.png": "31f68045e7ce16d6f38443654a1c22d431bc03e0d9b0402c7bd911532c12486f",
    "war_cartas7.png": "8df2cedb06784e97c7896d15689c15fb940d8f5ef7d5ed79672db0587d56c6b9",
    "war_cartas8.png": "bcfe2c7719b208012def366fe19895074cb727bdc32eb999211ee978bd5242bf",
}


def _document_reference(source: str, locator: str) -> SourceReference:
    return SourceReference(source=source, locator=locator)


def _asset_reference(filename: str, locator: str) -> SourceReference:
    return SourceReference(
        source=f"doc/images/{filename}",
        locator=locator,
        sha256=f"sha256:{_ASSET_SHA256[filename]}",
    )


_TERRITORIES: tuple[TerritoryCatalogEntry, ...] = (
    ("T01", "Austrália", "oceania", (), "triangle", "war_cartas1.png"),
    ("T02", "Nova Guiné", "oceania", (), "circle", "war_cartas1.png"),
    ("T03", "Sumatra", "oceania", (), "square", "war_cartas1.png"),
    ("T04", "Bornéu", "oceania", (), "square", "war_cartas1.png"),
    ("T05", "Alaska", "north_america", (), "triangle", "war_cartas1.png"),
    ("T06", "Groenlândia", "north_america", (), "circle", "war_cartas1.png"),
    ("T07", "Nova York", "north_america", (), "triangle", "war_cartas1.png"),
    ("T08", "Califórnia", "north_america", (), "square", "war_cartas1.png"),
    ("T09", "Vancouver", "north_america", (), "triangle", "war_cartas2.png"),
    ("T10", "Mackenzie", "north_america", (), "circle", "war_cartas2.png"),
    ("T11", "Labrador", "north_america", (), "square", "war_cartas2.png"),
    ("T12", "México", "north_america", (), "square", "war_cartas2.png"),
    ("T13", "Ottawa", "north_america", (), "circle", "war_cartas2.png"),
    ("T14", "África do Sul", "africa", (), "triangle", "war_cartas2.png"),
    ("T15", "Egito", "africa", (), "triangle", "war_cartas2.png"),
    ("T16", "Congo", "africa", (), "square", "war_cartas2.png"),
    ("T17", "Madagascar", "africa", (), "circle", "war_cartas3.png"),
    ("T18", "Argélia", "africa", (), "circle", "war_cartas3.png"),
    ("T19", "Sudão", "africa", (), "square", "war_cartas3.png"),
    ("T20", "Índia", "asia", (), "square", "war_cartas3.png"),
    ("T21", "Sibéria", "asia", (), "triangle", "war_cartas3.png"),
    ("T22", "Tchita", "asia", (), "triangle", "war_cartas3.png"),
    ("T23", "Vietnã", "asia", (), "triangle", "war_cartas3.png"),
    ("T24", "Japão", "asia", (), "square", "war_cartas3.png"),
    ("T25", "Argentina", "south_america", (), "square", "war_cartas4.png"),
    ("T26", "Brasil", "south_america", (), "circle", "war_cartas4.png"),
    ("T27", "Colômbia", "south_america", ("Venezuela",), "triangle", "war_cartas4.png"),
    ("T28", "Alemanha", "europe", (), "circle", "war_cartas4.png"),
    ("T29", "França", "europe", (), "square", "war_cartas4.png"),
    ("T30", "Inglaterra", "europe", (), "circle", "war_cartas4.png"),
    ("T31", "Suécia", "europe", (), "circle", "war_cartas4.png"),
    ("T32", "Islândia", "europe", (), "triangle", "war_cartas4.png"),
    ("T33", "Polônia", "europe", (), "square", "war_cartas6.png"),
    ("T34", "Moscou", "europe", (), "triangle", "war_cartas6.png"),
    ("T35", "China", "asia", (), "circle", "war_cartas7.png"),
    ("T36", "Dudinka", "asia", (), "circle", "war_cartas7.png"),
    ("T37", "Mongólia", "asia", (), "circle", "war_cartas7.png"),
    ("T38", "Omsk", "asia", (), "square", "war_cartas7.png"),
    ("T39", "Oriente Médio", "asia", (), "square", "war_cartas7.png"),
    ("T40", "Vladvostok", "asia", (), "circle", "war_cartas7.png"),
    ("T41", "Aral", "asia", (), "triangle", "war_cartas7.png"),
    (
        "T42",
        "Bolívia",
        "south_america",
        ("Peru", "Chile"),
        "triangle",
        "war_cartas7.png",
    ),
)

_CONTINENTS: tuple[tuple[str, str, int], ...] = (
    ("north_america", "América do Norte", 5),
    ("south_america", "América do Sul", 2),
    ("europe", "Europa", 5),
    ("africa", "África", 3),
    ("asia", "Ásia", 7),
    ("oceania", "Oceania", 2),
)

_ADJACENCY_PAIRS: tuple[tuple[str, str], ...] = (
    ("T01", "T02"),
    ("T01", "T03"),
    ("T01", "T04"),
    ("T02", "T04"),
    ("T03", "T20"),
    ("T04", "T23"),
    ("T05", "T09"),
    ("T05", "T10"),
    ("T05", "T40"),
    ("T06", "T10"),
    ("T06", "T11"),
    ("T06", "T32"),
    ("T07", "T08"),
    ("T07", "T11"),
    ("T07", "T12"),
    ("T07", "T13"),
    ("T08", "T09"),
    ("T08", "T12"),
    ("T08", "T13"),
    ("T09", "T10"),
    ("T09", "T13"),
    ("T10", "T13"),
    ("T11", "T13"),
    ("T12", "T27"),
    ("T14", "T16"),
    ("T14", "T17"),
    ("T14", "T19"),
    ("T15", "T18"),
    ("T15", "T19"),
    ("T15", "T29"),
    ("T15", "T33"),
    ("T15", "T39"),
    ("T16", "T18"),
    ("T16", "T19"),
    ("T17", "T19"),
    ("T18", "T19"),
    ("T18", "T26"),
    ("T18", "T29"),
    ("T20", "T23"),
    ("T20", "T35"),
    ("T20", "T39"),
    ("T20", "T41"),
    ("T21", "T22"),
    ("T21", "T36"),
    ("T21", "T40"),
    ("T22", "T35"),
    ("T22", "T36"),
    ("T22", "T37"),
    ("T22", "T40"),
    ("T23", "T35"),
    ("T24", "T35"),
    ("T24", "T40"),
    ("T25", "T26"),
    ("T25", "T42"),
    ("T26", "T27"),
    ("T26", "T42"),
    ("T27", "T42"),
    ("T28", "T29"),
    ("T28", "T30"),
    ("T28", "T33"),
    ("T29", "T30"),
    ("T29", "T33"),
    ("T30", "T31"),
    ("T30", "T32"),
    ("T31", "T34"),
    ("T33", "T34"),
    ("T33", "T39"),
    ("T34", "T38"),
    ("T34", "T39"),
    ("T34", "T41"),
    ("T35", "T37"),
    ("T35", "T38"),
    ("T35", "T40"),
    ("T35", "T41"),
    ("T36", "T37"),
    ("T36", "T38"),
    ("T37", "T38"),
    ("T38", "T41"),
    ("T39", "T41"),
)

_OBJECTIVES: tuple[tuple[str, str, str], ...] = (
    (
        "W1-OBJ-01",
        "Destruir totalmente os exércitos vermelhos; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-02",
        "Destruir totalmente os exércitos azuis; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-03",
        "Destruir totalmente os exércitos brancos; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas6.png",
    ),
    (
        "W1-OBJ-04",
        "Destruir totalmente os exércitos pretos; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas6.png",
    ),
    (
        "W1-OBJ-05",
        "Destruir totalmente os exércitos amarelos; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas6.png",
    ),
    (
        "W1-OBJ-06",
        "Destruir totalmente os exércitos verdes; aplica-se a conversão para 24 territórios descrita na carta.",
        "war_cartas6.png",
    ),
    (
        "W1-OBJ-07",
        "Conquistar na totalidade a América do Norte e a Oceania.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-08",
        "Conquistar na totalidade a Ásia e a América do Sul.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-09",
        "Conquistar na totalidade a Ásia e a África.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-10",
        "Conquistar na totalidade a América do Norte e a África.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-11",
        "Conquistar na totalidade a Europa, a América do Sul e mais um terceiro continente à escolha.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-12",
        "Conquistar na totalidade a Europa, a Oceania e mais um terceiro continente à escolha.",
        "war_cartas5.png",
    ),
    (
        "W1-OBJ-13",
        "Conquistar 24 territórios à sua escolha.",
        "war_cartas8.png",
    ),
    (
        "W1-OBJ-14",
        "Conquistar 18 territórios e ocupar cada um deles com pelo menos dois exércitos.",
        "war_cartas8.png",
    ),
)

_WILDCARD_CARD_IDS = ("JOKER-01", "JOKER-02")

_territories = tuple(
    TerritoryDefinition(
        territory_id=territory_id,
        canonical_name=canonical_name,
        continent_id=continent_id,
        aliases=aliases,
        card_symbol=card_symbol,
        source_references=(
            _document_reference(
                "doc/war_board_spec_v1.0.md",
                f"Catálogo de territórios: {territory_id}",
            ),
            _asset_reference(
                card_sheet,
                f"Carta territorial: {territory_id} ({canonical_name})",
            ),
        ),
    )
    for territory_id, canonical_name, continent_id, aliases, card_symbol, card_sheet in _TERRITORIES
)

_continents = tuple(
    ContinentDefinition(
        continent_id=continent_id,
        name=name,
        reinforcement_bonus=reinforcement_bonus,
        source_references=(
            _document_reference(
                "doc/war_board_spec_v1.0.md",
                f"Tabela de bônus continentais: {continent_id}",
            ),
            _asset_reference(
                "mapa_war_classico.webp",
                f"Tabuleiro: continente {name}",
            ),
        ),
    )
    for continent_id, name, reinforcement_bonus in _CONTINENTS
)

_adjacencies = tuple(
    Adjacency(
        first_territory_id=first,
        second_territory_id=second,
        source_references=(
            _document_reference(
                "doc/war_adjacency_review_v1.0.md",
                f"Adjacência não direcionada: {first} — {second}",
            ),
            _document_reference(
                "doc/war_adjacency_candidate_v1.0.json",
                f"adjacency.{first} -> {second}; adjacency.{second} -> {first}",
            ),
        ),
    )
    for first, second in _ADJACENCY_PAIRS
)

_objectives = tuple(
    ObjectiveDefinition(
        objective_id=objective_id,
        description=description,
        certified_rule_ids=(objective_id,),
        source_references=(
            _document_reference(
                "doc/war_rules_traceability_v1.0.md",
                f"Catálogo de objetivos: {objective_id}",
            ),
            _asset_reference(
                card_sheet,
                f"Carta de objetivo: {objective_id}",
            ),
        ),
    )
    for objective_id, description, card_sheet in _OBJECTIVES
)

WAR1_BOARD_SPEC = BoardSpec(
    contract_type="board_spec",
    board_id="war1-classic",
    version="1.0.0",
    map=MapSpec(
        map_id="war1-classic",
        version="1.0.0",
        territories=_territories,
        continents=_continents,
        adjacencies=_adjacencies,
        source_references=(
            _document_reference(
                "doc/war_board_spec_v1.0.md",
                "Catálogo T01–T42 e composição dos continentes",
            ),
            _document_reference(
                "doc/war_adjacency_review_v1.0.md",
                "Lista de adjacências fornecida/corrigida pelo usuário",
            ),
        ),
    ),
    territory_deck=DeckSpec(
        deck_id="war1-classic-territory-deck",
        version="1.0.0",
        card_ids=tuple(entry[0] for entry in _TERRITORIES) + _WILDCARD_CARD_IDS,
        source_references=(
            _document_reference(
                "doc/war_board_spec_v1.0.md",
                "Baralho: 42 cartas territoriais e 2 curingas",
            ),
            _document_reference(
                "doc/war_rules_traceability_v1.0.md",
                "W1-COMPONENTS-01 e W1-CARD-SYMBOLS-01",
            ),
            _asset_reference("war_cartas6.png", "Dois curingas"),
        ),
    ),
    objective_deck=DeckSpec(
        deck_id="war1-classic-objective-deck",
        version="1.0.0",
        card_ids=tuple(objective[0] for objective in _OBJECTIVES),
        source_references=(
            _document_reference(
                "doc/war_rules_traceability_v1.0.md",
                "W1-OBJECTIVE-CATALOG-01",
            ),
            _document_reference(
                "doc/war_setup_spec_v1.0.md",
                "14 cartas de objetivo e sorteio sem reposição",
            ),
        ),
    ),
    objectives=_objectives,
    wildcard_card_ids=_WILDCARD_CARD_IDS,
    source_references=(
        _document_reference(
            "doc/war_board_spec_v1.0.md",
            "Inventário canônico do tabuleiro e componentes",
        ),
        _document_reference(
            "doc/war_rules_traceability_v1.0.md",
            "Precedência das fontes físicas e esclarecimentos normativos",
        ),
    ),
)

__all__ = ["WAR1_BOARD_SPEC"]
