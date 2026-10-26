import json
import subprocess
import sys
from pathlib import Path


def _engine_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts" / "stress_engine.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the engine root containing scripts/.")


ENGINE_ROOT = _engine_root()


def _run_stress_cli(workers: int) -> dict[str, object]:
    result = subprocess.run(
        (
            sys.executable,
            str(ENGINE_ROOT / "scripts" / "stress_engine.py"),
            "--games",
            "2",
            "--max-transitions-per-game",
            "1",
            "--seed",
            "42",
            "--workers",
            str(workers),
        ),
        cwd=ENGINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report: object = json.loads(result.stdout)
    assert isinstance(report, dict)
    return report


def test_stress_cli_reports_truncations_and_replay_checks() -> None:
    report = _run_stress_cli(workers=1)
    assert report["attempted_games"] == 2
    assert report["completed_games"] == 0
    assert report["truncated_games"] == 2
    assert report["crashed_games"] == 0
    assert report["transitions"] == 2
    assert report["invariant_violations"] == 0
    assert report["replay_checks"] == 2
    assert report["replay_divergences"] == 0
    assert report["phase_transition_counts"] == {"initial_placement": 2}

def test_parallel_stress_campaign_is_deterministic() -> None:
    serial_report = _run_stress_cli(workers=1)
    parallel_report = _run_stress_cli(workers=2)
    serial_report.pop("elapsed_seconds")
    parallel_report.pop("elapsed_seconds")

    assert serial_report["workers"] == 1
    assert parallel_report["workers"] == 2
    serial_report["workers"] = parallel_report["workers"]
    assert serial_report == parallel_report
