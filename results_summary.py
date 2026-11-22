import csv
import sys
from collections import defaultdict
from pathlib import Path

def summarize_results(csv_path="results.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: {path} not found.")
        return

    grouped = defaultdict(list)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("Model"):
                continue
            key = (
                row["Model"].strip(),
                int(row["Rows"]),
                int(row["Columns"]),
                int(row["Mines"]),
            )
            grouped[key].append(row)

    if not grouped:
        print(f"No valid game records found in {path}.")
        return

    print(f"Summary of {path} ({sum(len(v) for v in grouped.values())} total games)\n" + "=" * 78)

    for (model, rows, cols, mines), records in grouped.items():
        total_games = len(records)
        wins = [r for r in records if r["Outcome"].strip().lower() == "victory"]
        losses = [r for r in records if r["Outcome"].strip().lower() != "victory"]

        win_count = len(wins)
        win_rate = (win_count / total_games) * 100.0

        all_runtimes = [float(r["Runtime (Seconds)"]) for r in records]
        win_runtimes = [float(r["Runtime (Seconds)"]) for r in wins]
        loss_runtimes = [float(r["Runtime (Seconds)"]) for r in losses]

        turns = [int(r["Turns Taken"]) for r in records]
        csp_moves = [int(r["CSP Moves"]) for r in records]
        model_moves = [int(r["DQN Moves"]) for r in records]
        popups = sum(1 for r in records if r["New Record Popup"].strip().lower() == "true")

        total_csp = sum(csp_moves)
        total_model = sum(model_moves)
        total_actions = total_csp + total_model
        csp_share = (total_csp / total_actions * 100.0) if total_actions > 0 else 0.0
        model_share = (total_model / total_actions * 100.0) if total_actions > 0 else 0.0

        print(f"Model: {model} | Board: {rows}x{cols} ({mines} mines)")
        print(f"  Games Played      : {total_games} (Wins: {win_count}, Losses: {len(losses)})")
        print(f"  Win Rate          : {win_rate:.2f}%")
        print(f"  Avg Runtime (All) : {sum(all_runtimes) / total_games:.2f}s (Min: {min(all_runtimes):.2f}s, Max: {max(all_runtimes):.2f}s)")
        if win_runtimes:
            print(f"  Avg Runtime (Win) : {sum(win_runtimes) / len(win_runtimes):.2f}s (Min: {min(win_runtimes):.2f}s, Max: {max(win_runtimes):.2f}s)")
        if loss_runtimes:
            print(f"  Avg Runtime (Loss): {sum(loss_runtimes) / len(loss_runtimes):.2f}s (Min: {min(loss_runtimes):.2f}s, Max: {max(loss_runtimes):.2f}s)")
        print(f"  Avg Turns / Game  : {sum(turns) / total_games:.2f}")
        print(f"  Avg CSP Moves     : {sum(csp_moves) / total_games:.2f} ({csp_share:.1f}% of all moves)")
        print(f"  Avg Model Guesses : {sum(model_moves) / total_games:.2f} ({model_share:.1f}% of all moves)")
        print(f"  Record Popups     : {popups}")
        print("-" * 78)


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    summarize_results(target_csv)
