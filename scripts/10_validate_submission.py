"""Script 10: Validate submission files against competition format rules."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching", default="outputs/matching_results.tsv", help="Matching results file")
    parser.add_argument("--candidate", default="outputs/candidate_pairs.tsv", help="Candidate pairs file")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory containing test_source*.tsv")
    args = parser.parse_args()

    validator_script = Path("utils/validate_submission.py")
    if not validator_script.exists():
        print(f"Error: Validator script {validator_script} not found.")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching",
        args.matching,
        "--candidate",
        args.candidate,
        "--test-dir",
        args.test_dir,
    ]
    print(f"[10_validate_submission] Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
