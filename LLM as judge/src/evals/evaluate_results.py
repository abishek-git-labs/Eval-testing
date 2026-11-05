import argparse
import json
from pathlib import Path

from src.config import RESULTS_DIR


def normalize(text):
    return " ".join(text.casefold().split())


parser = argparse.ArgumentParser()
parser.add_argument("--results-file")
args = parser.parse_args()

if args.results_file:
    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = RESULTS_DIR / results_path
else:
    result_files = list(RESULTS_DIR.glob("eval_results_*.json"))
    results_path = (
        max(result_files, key=lambda path: path.stat().st_mtime)
        if result_files
        else RESULTS_DIR / "eval_results.json"
    )
results = json.loads(
    results_path.read_text(encoding="utf-8")
)

if not results:
    raise ValueError("The results file is empty.")

matches = 0

for result in results:
    passed = (
        normalize(result["model_answer"])
        == normalize(result["reference_answer"])
    )
    matches += int(passed)

    print(
        f"{result['id']}: "
        f"{'MATCH' if passed else 'DIFFERENT WORDING OR CONTENT'}"
    )

print(f"\nExact matches: {matches}/{len(results)}")
print(f"Exact-match rate: {matches / len(results):.1%}")