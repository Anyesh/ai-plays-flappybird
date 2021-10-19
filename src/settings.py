from pathlib import Path
from rich.console import Console

BASEDIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASEDIR / "artifacts"

MODEL_NAME = "model_v1"


console = Console()
