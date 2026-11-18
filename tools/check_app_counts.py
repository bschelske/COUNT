"""
Check a built COUNT app against the synthetic videos' answer keys (used by .github/workflows/build-app.yml).

    python tools/make_synthetic_videos.py synthetic_videos
    <the app> --count synthetic_videos/*.avi --out app_results --set count_line_x=0 --set outlet_divider_y=0 ...
    python tools/check_app_counts.py synthetic_videos app_results

Compares the number of cells counted at each outlet with each video's answer key, and exits with an error if
any video has no results or different counts.
"""
import csv
import os
import sys


def outlet_counts(path, column="DEP_response"):
    """(bottom / nDEP, top / pDEP) counts in a results or answer key CSV file."""
    with open(path, newline="") as file:
        responses = [row[column] == "True" for row in csv.DictReader(file)]
    return len(responses) - sum(responses), sum(responses)


def main(videos_folder, results_folder):
    keys = sorted(f for f in os.listdir(videos_folder) if f.endswith("_answer_key.csv"))
    if not keys:
        print(f"No answer keys in {videos_folder}")
        return 1
    failed = False
    for key in keys:
        name = key[:-len("_answer_key.csv")]
        expected = outlet_counts(os.path.join(videos_folder, key))
        results = os.path.join(results_folder, f"{name}_results.csv")
        if not os.path.exists(results):
            print(f"{name}: no results file (expected {results})")
            failed = True
            continue
        counted = outlet_counts(results)
        ok = counted == expected
        failed |= not ok
        print(f"{name}: counted {counted[0]} nDEP / {counted[1]} pDEP, answer key {expected[0]} / {expected[1]}"
              f" {'OK' if ok else 'DIFFERENT'}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
