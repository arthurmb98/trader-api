from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "studies" / "results"
WEB_DIR = ROOT / "web"
WEB_DIST = WEB_DIR / "dist"
