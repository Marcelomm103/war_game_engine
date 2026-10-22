# war_game_engine
Repository containing the game engine of the boardgame war (risk)

## Install

The package uses a `src` layout and is imported as `war_game_engine`.
Install the runtime package and its test tools from this directory:

```bash
python -m pip install -e ".[test,quality]"
```

NumPy and Pydantic are runtime dependencies for numerical work and validated
contracts. Importing `war_game_engine` does not import graphical or
neural-network libraries.

Ruff, mypy, and mutmut are pinned in the `quality` extra. The engine test suite
covers canonical data, state projections, setup, action validation, random
streams, snapshots, and replay foundations.

The canonical transport models are in `war_game_engine.contracts`. Regenerate
their JSON Schema from this directory with:

```bash
python3 scripts/export_contract_schema.py
```

The validated WAR 1 classic board catalog is available as
`war_game_engine.WAR1_BOARD_SPEC`. It includes territories, continents,
adjacencies, territory and objective decks, and source references for each
catalog entry. Invalid IDs, graph references, duplicate cards or objectives,
and incomplete provenance are rejected during model validation.

## Phase 2 engine foundations

`war_game_engine.setup.prepare_game()` creates a reproducible initial game for
3–6 `SetupPlayer`s. It filters elimination objectives by participating colors,
records the tie-break rounds for distributor selection, deals territories,
assigns initial reinforcement pools, and returns the initial placement order
and an internal setup transcript:

```python
from war_game_engine.setup import SetupPlayer, prepare_game

setup = prepare_game(
    "game-1",
    (
        SetupPlayer(player_id="player-1", color="red"),
        SetupPlayer(player_id="player-2", color="blue"),
        SetupPlayer(player_id="player-3", color="green"),
    ),
    seed=42,
)
full_state = setup.state
```

`war_game_engine.randomness.RandomStreams` keeps preparation, objectives,
cards, combat, and per-policy PCG64 streams independent and serializable.
Their root seed and generator states live only in `FullGameState`. Use
`to_public_game_state(full_state)` and
`observe_player(full_state, player_id)` from `war_game_engine.state` to produce
public or player-scoped data; these
projections omit the RNG state and other players' private data.

`war_game_engine.engine` exposes pure `validate_action()`, `legal_actions()`,
and `apply_action()` APIs for territory claims and initial/reinforcement
placement. Other turn phases fail explicitly as unsupported until Phase 3.
`war_game_engine.replay` provides canonical state hashes, versioned snapshots,
state/RNG restoration, and separate full and sanitized public replay contracts.

Independent semantic versions and certificate invalidation are defined in
`war_game_engine.versioning`. A changed version invalidates only certificates
whose scope depends on that dimension; historical manifests are retained.
