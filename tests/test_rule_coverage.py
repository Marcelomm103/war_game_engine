import re
from pathlib import Path

import pytest


def _workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "doc" / "war_rules_traceability_v1.0.md").is_file():
            return candidate
    raise RuntimeError("Could not locate the workspace root containing doc/.")


RULE_MATRIX_PATH = _workspace_root() / "doc" / "war_rules_traceability_v1.0.md"
ENGINE_SCOPE_EXCLUSIONS = frozenset({"W1-TROOP-DENOMINATION-01"})


def test_every_engine_rule_id_has_a_traceable_test(
    request: pytest.FixtureRequest,
) -> None:
    if (
        request.config.getoption("keyword")
        or request.config.getoption("markexpr")
        or any(Path(argument).suffix == ".py" for argument in request.config.args)
    ):
        pytest.skip("Rule-ID coverage requires the complete engine test collection.")

    matrix = RULE_MATRIX_PATH.read_text(encoding="utf-8")
    documented_rule_ids = set(re.findall(r"`(W1-[A-Z0-9-]+)`", matrix))
    covered_rule_ids = {
        rule_id
        for item in request.session.items
        for marker in item.iter_markers(name="rule_id")
        for rule_id in marker.args
        if isinstance(rule_id, str)
    }

    assert ENGINE_SCOPE_EXCLUSIONS <= documented_rule_ids
    assert covered_rule_ids <= documented_rule_ids
    missing_rule_ids = sorted(
        documented_rule_ids - covered_rule_ids - ENGINE_SCOPE_EXCLUSIONS
    )
    assert not missing_rule_ids, (
        "Traceability rule IDs without executable engine tests: "
        f"{', '.join(missing_rule_ids)}."
    )
