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

Ruff, mypy, and mutmut are pinned in the `quality` extra. Mutation campaigns
become meaningful when engine behavior and its tests are added.

The canonical transport models are in `war_game_engine.contracts`. Regenerate
their JSON Schema from this directory with:

```bash
python3 scripts/export_contract_schema.py
```

Independent semantic versions and certificate invalidation are defined in
`war_game_engine.versioning`. A changed version invalidates only certificates
whose scope depends on that dimension; historical manifests are retained.
