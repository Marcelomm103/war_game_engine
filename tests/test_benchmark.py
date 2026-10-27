import json
import subprocess
import sys
from pathlib import Path


def _engine_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts" / "benchmark_engine.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the engine root containing scripts/.")


ENGINE_ROOT = _engine_root()


def test_benchmark_cli_reports_measured_workloads_and_separates_truncations(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "benchmark.json"
    result = subprocess.run(
        (
            sys.executable,
            str(ENGINE_ROOT / "scripts" / "benchmark_engine.py"),
            "--games",
            "1",
            "--max-transitions-per-game",
            "1",
            "--iterations",
            "2",
            "--profile-top",
            "2",
            "--seed",
            "20261023",
            "--output",
            str(report_path),
        ),
        cwd=ENGINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == saved_report
    assert stdout_report["schema_version"] == "1.0"

    campaign = stdout_report["game_campaign"]
    assert campaign["attempted_games"] == 1
    assert campaign["completed_games"] == 0
    assert campaign["truncated_games"] == 1
    assert campaign["transitions"] == 1
    assert campaign["invariant_violations"] == 0
    assert campaign["replay_checks"] == 1
    assert campaign["replay_divergences"] == 0
    assert campaign["transitions_per_second"] > 0
    assert campaign["completed_games_per_second"] == 0

    microbenchmarks = stdout_report["microbenchmarks"]
    assert microbenchmarks["seed"] == 20261023
    assert microbenchmarks["iterations"] == 2
    assert set(microbenchmarks["cloning"]) == {
        "pydantic_shallow_model_copy",
        "pydantic_deep_model_copy",
        "state_model_validate_from_pre_dumped_dict",
    }
    assert microbenchmarks["memory"]["canonical_state_json_bytes"] > 0
    assert set(microbenchmarks["validation"]) == {
        "full_state_validation_from_pre_dumped_dict",
        "single_legal_attack_action",
    }

    phases = microbenchmarks["action_generation_by_phase"]
    assert {
        "territory_claim_synthetic",
        "initial_placement",
        "reinforcement",
        "reinforcement_with_five_cards_synthetic",
        "reinforcement_placement",
        "attack",
        "occupation_synthetic",
        "movement",
        "card_purchase_synthetic",
        "card_discard_synthetic",
        "turn_end",
    }.issubset(phases)
    reference_report = json.loads(
        (
            ENGINE_ROOT
            / "benchmarks"
            / "phase-4.7-reference-2026-10-27.json"
        ).read_text(encoding="utf-8")
    )
    reference_phases = reference_report["microbenchmarks"][
        "action_generation_by_phase"
    ]
    assert {
        phase: measurement["action_signature_sha256"]
        for phase, measurement in phases.items()
    } == {
        phase: measurement["action_signature_sha256"]
        for phase, measurement in reference_phases.items()
    }
    for measurement in phases.values():
        assert measurement["actions_per_call"] > 0
        assert measurement["latency"]["median_us"] > 0
        assert len(measurement["top_cumulative_profile_functions"]) <= 2
