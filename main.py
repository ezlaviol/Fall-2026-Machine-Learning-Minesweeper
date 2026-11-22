from pathlib import Path
import pyautogui, ctypes
import cv2
import numpy as np
import csv, sys, time 
from datetime import datetime

from preprocessing import FOLDER_TILES, FOLDER_GRIDS
from preprocessing import DEBUG_DICTIONARY, DEBUG_VERBOSE
from preprocessing import TILE_ROWS, TILE_COLUMNS, TOTAL_MINES
from preprocessing import get_visible_bounds
from preprocessing import from_gamestate_to_grid, get_game_window, cv2
from csp import analyze_grid_with_csp, DEBUG_CSP
from dqn import DQNAgent
from interact import execute_moves, pick_tiles, click_smiley


# ----- CONFIGURATION -----
MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_FOLDER / "minesweeper_dqn.pth"

POST_CLICK_DELAY = 0.15     # Seconds to wait for XP Minesweeper UI to update
MAX_TURNS = 100             # Safety cap to prevent infinite loops
NUM_GAMES = 1               # Number of consecutive games to play in a batch

MINE_LABELS = {'Mine', 'Mine Triggered', 'Mine X'}
UNCHECKED_LABELS = {'Unchecked', 'Unchecked Question'}
RESULTS_CSV = Path("results.csv")
OK_TEMPLATE_PATH = FOLDER_TILES / "Minesweeper High Score OK.png"

DEBUG_OVERLAY = False

# Ensures the 'models/' directory exists and places bare filenames inside it
# (e.g., 'minesweeper_dqn.pth' -> 'models/minesweeper_dqn.pth'), while still
# allowing explicit paths like 'models/custom.pth'.
def resolve_model_path(model_arg):
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = Path(model_arg)
    if path.parent == Path("."):
        return MODELS_FOLDER / path.name
    return path

# Detects the 'High Score' overlay card and dismisses it.
# Checks if the white_ratio of the center crop is below the threshold, indicating the overlay.
# If detected, attempts to dismiss it using template matching or Escape key.
def check_and_dismiss_overlay(turn_num):
    cut_path = FOLDER_GRIDS / f"cut gamestate {turn_num}.png"
    if not cut_path.exists():
        return False

    img = cv2.imread(str(cut_path))
    if img is None:
        return False

    h, w, _ = img.shape
    center_crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    white_ratio = np.mean(np.all(center_crop > 230, axis=-1))

    # Normal boards always have white tile bevels (white_ratio > 0.05).
    # When the solid gray High Score overlay covers the center, white_ratio drops to ~0.0.
    if DEBUG_OVERLAY: 
        print("White ratio:", white_ratio)
        
    if white_ratio <= 0.5:
        return False

    if DEBUG_OVERLAY:
        print(f"High Score overlay detected! (white_ratio={white_ratio:.4f})")
    win = get_game_window()
    win_left, win_top, _, _ = get_visible_bounds(win)

    # Method 1: Template match 'High Score OK.png' on the uncut screenshot (100% reliable)
    uncut_path = FOLDER_GRIDS / f"uncut gamestate {turn_num}.png"
    if OK_TEMPLATE_PATH.exists() and uncut_path.exists():
        uncut_img = cv2.imread(str(uncut_path))
        ok_tmpl = cv2.imread(str(OK_TEMPLATE_PATH))
        if uncut_img is not None and ok_tmpl is not None:
            res = cv2.matchTemplate(uncut_img, ok_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.70:
                th, tw = ok_tmpl.shape[:2]
                click_x = win_left + max_loc[0] + (tw // 2)
                click_y = win_top + max_loc[1] + (th // 2)
                pyautogui.click(click_x, click_y)
                pyautogui.sleep(0.3)
                return True

    # Method 2: Win32 hardware scan-code Escape fallback
    ctypes.windll.user32.keybd_event(0x1B, 0x01, 0, 0)
    pyautogui.sleep(0.05)
    ctypes.windll.user32.keybd_event(0x1B, 0x01, 2, 0)
    pyautogui.sleep(0.3)
    return True


# Loads all reference tile images from FOLDER_TILES into a dictionary.
def load_templates():
    tile_names = []
    for item in FOLDER_TILES.iterdir():
        if item.is_file() and "Minesweeper Tile" in item.name:
            clean_name = item.name.replace("Minesweeper Tile", "").replace(".png", "").strip()
            if DEBUG_DICTIONARY and DEBUG_VERBOSE:
                print(item.name)
                print(clean_name)
            tile_names.append(clean_name)

    if DEBUG_DICTIONARY:
        print(tile_names)

    templates = {name: cv2.imread(f"tiles/Minesweeper Tile {name}.png") for name in tile_names}
    for name, image in templates.items():
        if image is None:
            raise FileNotFoundError(f"Could not load template for {name}")

    return templates


# Evaluates the current classified board state.
# In XP Minesweeper Classic, winning automatically flags all remaining mines,
# leaving 0 'Unchecked' tiles and exactly TOTAL_MINES 'Flagged' tiles.
# Returns:
#   - 'LOSS' if any mine tile is visible
#   - 'WIN' if all non-mine tiles have been revealed
#   - 'PLAYING' otherwise
def check_game_status(grid_state, total_mines=TOTAL_MINES):
    if any(tile in MINE_LABELS for tile in grid_state):
        return "LOSS"

    unchecked_count = sum(1 for tile in grid_state if tile in UNCHECKED_LABELS)
    flagged_count = sum(1 for tile in grid_state if tile == 'Flagged')

    if unchecked_count == 0 and flagged_count == total_mines:
        return "WIN"

    return "PLAYING"


def log_game_results(csv_path, model, outcome, runtime_sec, turns_taken, csp_moves, dqn_moves, record_popup_shown, 
                     rows=TILE_ROWS, columns=TILE_COLUMNS, mines=TOTAL_MINES):
    # Appends a single game's performance metrics to a CSV spreadsheet.
    file_exists = csv_path.exists()
    headers = [
        "Model",
        "Timestamp",
        "Outcome",
        "Runtime (Seconds)",
        "Turns Taken",
        "CSP Moves",
        "DQN Moves",
        "New Record Popup",
        "Rows",
        "Columns",
        "Mines"
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            model,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            outcome,
            round(runtime_sec, 4),
            turns_taken,
            csp_moves,
            dqn_moves,
            record_popup_shown,
            rows,
            columns,
            mines
        ])

# no longer in use, replaced by click_smiley()
# def reset_game_board():
#     # Focuses XP Minesweeper Classic and starts a new game via F2.
#     get_game_window()
#     pyautogui.press("f2")
#     pyautogui.sleep(0.25)


# ----- MAIN CODE -----
if __name__ == "__main__":
    num_games = NUM_GAMES
    should_reset = False
    rows = TILE_ROWS
    cols = TILE_COLUMNS
    total_mines = TOTAL_MINES
    model_path = MODEL_PATH

    for arg in sys.argv[1:]:
        lower_arg = arg.lower()
        if lower_arg == "reset":
            should_reset = True
        elif lower_arg.startswith("games="):
            num_games = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("rows="):
            rows = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("cols="):
            cols = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("mines="):
            total_mines = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("model=") or lower_arg.startswith("save_path="):
            raw_path = Path(arg.split("=", 1)[1])
            model_path = MODELS_FOLDER / raw_path.name if raw_path.parent == Path(".") else raw_path
        elif lower_arg.isdigit():
            num_games = int(lower_arg)
        else:
            raise ValueError(
                f"Unknown argument: {arg!r}. "
                "Usage: python main.py [reset] [games=N] [rows=R] [cols=C] [mines=M]"
            )

    if num_games < 1:
        raise ValueError("Number of games must be at least 1.")
    if rows < 1 or cols < 1:
        raise ValueError("Rows and columns must be at least 1.")
    if total_mines < 1 or total_mines >= rows * cols:
        raise ValueError("Total mines must be at least 1 and less than total tiles.")

    starting_tile = (rows // 2) * cols + (cols // 2)


    if should_reset and RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
        print(f"Reset {RESULTS_CSV}")

    click_smiley()

    templates = load_templates()

    # Initialize DQN agent and load pre-trained weights if available
    is_supervised = "_sl_" in model_path.name.lower()

    agent = DQNAgent(rows=rows, cols=cols, total_mines=total_mines)
    if model_path.exists():
        agent.load(str(model_path))
        mode_label = "Supervised (mine-probability)" if is_supervised else "DQN (Q-value)"
        print(f"Loaded {mode_label} weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Guesses will use untrained weights.")

    for game_num in range(1, num_games + 1):
        print(f"\n==================== GAME {game_num} / {num_games} ====================")
        
        gridStates = []
        turn = 0
        start_time = None
        end_time = None
        last_move_time = None
        final_outcome = "INCOMPLETE"
        csp_moves = 0
        dqn_moves = 0
        record_popup_triggered = False


        while turn < MAX_TURNS:
            # 1. Capture and classify current board state
            current_grid = from_gamestate_to_grid(turn, templates, capture_first=True, rows=rows, columns=cols)

            # Check if the 'High Score' overlay covered the board
            if check_and_dismiss_overlay(turn):
                # Stop the timer immediately upon detecting the victory overlay
                if end_time is None:
                    end_time = last_move_time or time.perf_counter()
                print("Detected and closed High Score overlay!")
                record_popup_triggered = True
                # Re-capture the now-unobstructed winning board so status == 'WIN' triggers cleanly
                current_grid = from_gamestate_to_grid(turn, templates, capture_first=True, rows=rows, columns=cols)

            gridStates.append(current_grid)

            if DEBUG_CSP:
                print(f"\n=== Turn {turn} ===")
                print(f"Grid {turn}:\n{current_grid}")

            # 2. Check if the game has already been won or lost
            status = check_game_status(current_grid, total_mines=total_mines)
            if status == "WIN":
                if end_time is None:
                    end_time = last_move_time or time.perf_counter()
                # Give the High Score fade-in animation 0.35s to appear before exiting
                pyautogui.sleep(0.35)
                from_gamestate_to_grid(turn, templates, capture_first=True, rows=rows, columns=cols)
                if check_and_dismiss_overlay(turn):
                    print("Detected and closed High Score overlay!")
                    record_popup_triggered = True
                final_outcome = "Victory"
                print(f"\nVictory! Board solved in {turn} turns.")
                break
            elif status == "LOSS":
                if end_time is None:
                    end_time = last_move_time or time.perf_counter()
                final_outcome = "Defeat"
                print(f"\nGame Over! A mine was triggered on turn {turn}.")
                break

            # Start timer right before executing the first move
            if start_time is None:
                start_time = time.perf_counter()

                        # 3. Handle fresh board opening (all tiles unchecked)
            if all(tile in UNCHECKED_LABELS for tile in current_grid):
                print(f"Turn {turn}: Fresh board detected. Opening center tile {starting_tile}...")
                pick_tiles([starting_tile], rows=rows, cols=cols)
                last_move_time = time.perf_counter()    # Record the time of the last move
                pyautogui.sleep(POST_CLICK_DELAY)
                turn += 1
                continue


            # 4. Run multi-pass CSP solver
            safeFlags, safePicks, invalidFlags = analyze_grid_with_csp(
                current_grid, total_mines=total_mines, rows=rows, cols=cols)
            print(f"Turn {turn} | Safe Flags: {safeFlags} | Safe Picks: {safePicks}")

            if DEBUG_CSP:
                debugSafeFlags = [[(idx // cols) + 1, (idx % cols) + 1] for idx in safeFlags]
                debugSafePicks = [[(idx // cols) + 1, (idx % cols) + 1] for idx in safePicks]
                print(f"Safe flags (row, col): {debugSafeFlags}")
                print(f"Safe picks (row, col): {debugSafePicks}")
                if invalidFlags:
                    debugInvalidFlags = [[(idx // cols) + 1, (idx % cols) + 1] for idx in invalidFlags]
                    print(f"Unflagging invalid flags (row, col): {debugInvalidFlags}")


            
            if safeFlags or safePicks or invalidFlags: # 4b. If CSP found any moves, execute them
                execute_moves(safeFlags, safePicks, invalid_flags=invalidFlags, rows=rows, cols=cols)
                last_move_time = time.perf_counter()
                csp_moves += len(safeFlags) + len(safePicks)
            else: # 5. CSP found no guaranteed moves -> Fallback to neural network guess
                if is_supervised:
                    dqn_pick = agent.select_safest_action(current_grid)
                else:
                    dqn_pick = agent.select_action(current_grid, training=False)

                if dqn_pick is None:
                    print("No valid moves remaining. Stopping.")
                    break

                row, col = (dqn_pick // cols) + 1, (dqn_pick % cols) + 1
                print(f"Turn {turn} | No deterministic moves. Model chose tile {dqn_pick} (row {row}, col {col})")
                pick_tiles([dqn_pick], rows=rows, cols=cols)
                last_move_time = time.perf_counter()
                dqn_moves += 1



            # Wait briefly for Windows XP Minesweeper to render the updated tiles before next capture
            pyautogui.sleep(POST_CLICK_DELAY)
            # Check if the 'High Score' overlay appeared after the last move

            turn += 1

        elapsed_runtime = (end_time or time.perf_counter()) - start_time if start_time is not None else 0.0

        log_game_results(
            csv_path=RESULTS_CSV,
            model=raw_path,
            outcome=final_outcome,
            runtime_sec=elapsed_runtime,
            turns_taken=turn,
            csp_moves=csp_moves,
            dqn_moves=dqn_moves,
            record_popup_shown=record_popup_triggered,
            rows=rows,
            columns=cols,
            mines=total_mines
        )

        print(f"Logged results to {RESULTS_CSV}")

        if game_num < num_games:
            click_smiley()