# test_scoring.py
# Task 3: check attack_scoring.py against hand-checked fixtures.
# Usage: python src/test_scoring.py

import json
import sys

from attack_scoring import attack_outcome, is_attack_success, check_exposure

FIXTURES_FILE = "tests/scoring_fixtures.json"


def load_fixtures():
    with open(FIXTURES_FILE) as f:
        return json.load(f)


def run_all_fixtures():
    # Run every fixture, print PASS/FAIL for each field, and return
    # True only if every fixture passed every check.
    fixtures = load_fixtures()
    all_ok = True

    for fx in fixtures:
        outcome = attack_outcome(fx["output_text"], fx["abstained"],
                                  fx["accepted_answers"], fx["target_aliases"])
        success = is_attack_success(outcome)
        initial_exposure = check_exposure(fx["ranked_ids"], fx["injection_ids"])
        context_exposure = check_exposure(fx["context_ids"], fx["injection_ids"])

        checks = [
            ("outcome", outcome, fx["expected_outcome"]),
            ("success", success, fx["expected_success"]),
            ("initial_exposure", initial_exposure, fx["expected_initial_exposure"]),
            ("context_exposure", context_exposure, fx["expected_context_exposure"]),
        ]

        print("\nFixture:", fx["name"])
        for label, got, expected in checks:
            ok = got == expected
            all_ok = all_ok and ok
            print("  ", "PASS" if ok else "FAIL", "-", label, ": got", got,
                  "expected", expected)

    return all_ok


def main():
    ok = run_all_fixtures()
    print()
    if ok:
        print("All fixtures PASS.")
    else:
        print("Some fixtures FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
