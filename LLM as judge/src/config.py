import os
from pathlib import Path

from dotenv import load_dotenv


SRC_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SRC_DIR.parent
DATASETS_DIR = SRC_DIR / "datasets"
RESULTS_DIR = SRC_DIR / "results"

load_dotenv(WORKSPACE_DIR / ".env")

EVAL_MODEL = os.getenv("EVAL_MODEL")
DEFAULT_PROFILE = "willow"
EVAL_PROFILES = {
    "willow": {
        "policy_file": "willow-pine-returns-policy.txt",
        "dataset_file": "eval_dataset.json",
    },
    "it-helpdesk": {
        "policy_file": "it-helpdesk-policy.txt",
        "dataset_file": "it-helpdesk-dataset.json",
    },
}
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
DEFAULT_MODEL = "qwen3-8b"
LOCAL_BASE_URL = "http://localhost:1234/v1"
LOCAL_API_KEY = "lm-studio"
API_TIMEOUT = 180
API_MAX_RETRIES = 0
