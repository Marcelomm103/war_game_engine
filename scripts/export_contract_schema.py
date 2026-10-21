import json
from pathlib import Path

from war_game_engine.contracts import contract_json_schema


def main() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "war_game_engine"
        / "schemas"
        / "contract-envelope-v1.json"
    )
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        json.dumps(contract_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
