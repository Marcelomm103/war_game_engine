import subprocess
import sys


def test_engine_import_does_not_load_vision_or_neural_frameworks() -> None:
    script = (
        "import importlib, sys; "
        "importlib.import_module('war_game_engine'); "
        "assert 'cv2' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
