# war_game_engine
Repository containing the game engine of the boardgame war (risk)

## Install

This package requires Python 3.11 or newer. CI and the aggregate development
container use Python 3.11.

The package uses a `src` layout and is imported as `war_game_engine`.
Install the runtime package and its test tools from this directory:

```bash
python -m pip install -e ".[test,quality]"
```

NumPy and Pydantic are runtime dependencies for numerical work and validated
contracts. Importing `war_game_engine` does not import graphical or
neural-network libraries.

Coverage.py, Ruff, mypy, and mutmut are pinned in the `quality` extra. The
engine test suite covers canonical data, state projections, setup, action
validation, random streams, snapshots, and replay foundations.

Run branch coverage on the engine tests and the mutation campaign from this
directory:

```bash
python -m coverage run --branch --source=src/war_game_engine -m pytest -q tests
python -m coverage report --include=src/war_game_engine/engine.py,src/war_game_engine/combat.py,src/war_game_engine/objectives.py,src/war_game_engine/reinforcements.py,src/war_game_engine/setup.py
mutmut run --max-children 4
mutmut results --all True
```

The standalone stress runner records completed games, truncations, crashes,
invariant failures, transition counts, and full-replay divergences separately:
the JSON report also includes transition counts by source phase.

```bash
python scripts/stress_engine.py --games 8 --max-transitions-per-game 30 --seed 42 --workers 4
```

Increase `--games` and `--max-transitions-per-game` for longer campaigns.
`--workers` runs independent deterministic game indices in parallel. Truncated
episodes are never counted as completed games; the report records them
separately from crashes and replay divergences. A game is truncated only when
it is still nonterminal after reaching the per-game transition limit. One
transition is one applied action or one resolved combat outcome. Initialization
or transition exceptions are counted as crashes instead, and no wall-clock
timeout is used to decide whether a game is truncated.

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
and `apply_action()` APIs for territory claims, initial placement, explicit
phase completion, and turn advancement. Completing initial placement starts
round two with the first player in the certified order. Turn advancement skips
eliminated players and increments the round when play wraps to the beginning
of that order. The phase IDs are typed; card purchase is reachable after
movement only when a conquest occurred in that turn, and finished states
cannot accept actions.

The Phase 3.1 machine is the foundation for the remaining turn rules. Phase 3.2
adds turn reinforcement pools, card-set validation and rewards, purchase after
conquest, and the distinct hand-limit discard zone. Phase 3.3 validates attack
choices before issuing a fixed-dice `ChanceRequest`, resolves one public
`ChanceOutcome` at a time, and enumerates exact loss distributions. A defeated
territory is handed to a pending occupation decision. Phase 3.4 validates that
transfer, records elimination attribution and objective fallback, and pauses
at `objective_check` before another action is offered. Movement and objective
evaluation are handled in their own roadmap steps. Phase 3.5 validates direct
allied movement, garrisons, and the per-army movement ledger, pausing for
objective checks after each transfer. Phase 3.6 adds typed evaluators for the
certified objective catalog, immediate checks, simultaneous-winner
adjudication without an invented tie-break, and winner-only objective reveal
in public state. Elimination hand transfer and random over-limit discards run
after that adjudication, so a winning attacker does not receive the cards.
`war_game_engine.replay` provides canonical state hashes, versioned snapshots,
state/RNG restoration, and separate full and sanitized public replay contracts.
Phase 3.7 distinguishes simulated outcomes from site and physical-board
observations. Chance request/event IDs are retained and deduplicated in state
and replay; observed dice cannot advance the engine RNG, and combat accepts
only the exact faces requested rather than inferred aggregate losses.
Phase 3.8 labels replay endings as official victory, individual elimination,
resignation, operational interruption, or training truncation. Non-victory
endings must match a non-winning state; an episode limit cannot create a
winner.

Independent semantic versions and certificate invalidation are defined in
`war_game_engine.versioning`. A changed version invalidates only certificates
whose scope depends on that dimension; historical manifests are retained.
