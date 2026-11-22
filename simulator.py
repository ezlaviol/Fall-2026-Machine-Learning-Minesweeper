import random, sys
from collections import deque
from pathlib import Path
from preprocessing import TILE_ROWS, TILE_COLUMNS, TOTAL_MINES
from csp import analyze_grid_with_csp
from dqn import DQNAgent
from dqn import GAMMA
from main import LOGS_FOLDER

MODEL_PATH = "minesweeper_dqn.pth"  # Path to the trained DQN model
MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
TRAINING_FOLDER = Path("logs/training")
TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)


# Ensures the 'models/' directory exists and places bare filenames inside it
# (e.g., 'minesweeper_dqn.pth' -> 'models/minesweeper_dqn.pth'), while still
# allowing explicit paths like 'models/custom.pth'.
def resolve_model_path(model_arg):
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = Path(model_arg)
    if path.parent == Path("."):
        return MODELS_FOLDER / path.name
    return path

# Fast Minesweeper environment matching XP Minesweeper Classic rules:
# Mines are placed AFTER the first click so the first move never triggers a mine.
# Flood-fill reveal for empty (checked) tiles.
class MinesweeperEnv:
    def __init__(self, rows=TILE_ROWS, cols=TILE_COLUMNS, num_mines=10):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.total_tiles = rows * cols
        self.reset()

    # Resets the board to all 'Unchecked' tiles before the first click.
    def reset(self):
        self.mines = set()
        self.mines_placed = False
        self.revealed = set()
        self.flagged = set()
        self.done = False
        self.won = False
        self.triggered_mine = None
        return self.get_grid_state()

    # Returns the list of neighboring tile indices for a given tile index.
    def _get_neighbors(self, idx):
        r, c = divmod(idx, self.cols)
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append(nr * self.cols + nc)
        return neighbors

    # Places mines randomly after the first click, ensuring the first clicked tile is safe.
    def _place_mines(self, safe_idx):
        candidates = [i for i in range(self.total_tiles) if i != safe_idx]
        self.mines = set(random.sample(candidates, self.num_mines))
        self.mines_placed = True

    # Returns the number of adjacent mines for a given tile index.
    def _adjacent_mine_count(self, idx):
        return sum(1 for n in self._get_neighbors(idx) if n in self.mines)

    # Places a flag on an unrevealed tile.
    def flag(self, idx):
        if not self.done and idx not in self.revealed:
            self.flagged.add(idx)

    # Returns the current grid state as a list of strings matching preprocessing.py format.
    # Left-clicks `action_idx`.
    # Returns: (next_grid_state, reward, done, info)
    def step(self, action_idx):
        if self.done:
            return self.get_grid_state(), 0.0, True, {"won": self.won}

        # Invalid or redundant click
        if action_idx in self.revealed or action_idx in self.flagged:
            return self.get_grid_state(), -0.5, False, {"won": False}

        # XP Minesweeper first-click safety
        if not self.mines_placed:
            self._place_mines(action_idx)

        # Hit a mine -> Game Over
        if action_idx in self.mines:
            self.done = True
            self.won = False
            self.triggered_mine = action_idx
            return self.get_grid_state(), -1.0, True, {"won": False}

        # Check if the clicked tile touches at least one already-revealed tile
        is_progress = any(n in self.revealed for n in self._get_neighbors(action_idx))

        # Flood-fill reveal starting from action_idx
        queue = deque([action_idx])
        while queue:
            curr = queue.popleft()
            if curr in self.revealed or curr in self.flagged:
                continue
            self.revealed.add(curr)

            if self._adjacent_mine_count(curr) == 0:
                for n in self._get_neighbors(curr):
                    if n not in self.revealed and n not in self.flagged:
                        queue.append(n)

        # Check win condition: all non-mine tiles revealed
        if len(self.revealed) == self.total_tiles - self.num_mines:
            self.done = True
            self.won = True
            return self.get_grid_state(), 1.0, True, {"won": True}

        # Reward frontier progress (+0.3), penalize isolated blind guesses (-0.2)
        reward = 0.3 if is_progress else -0.2
        return self.get_grid_state(), reward, False, {"won": False}

    # Returns the current grid state as a list of strings matching preprocessing.classify_grid().
    def get_grid_state(self):
        grid = []
        for i in range(self.total_tiles):
            if self.done and not self.won:
                # Reveal all mines on loss just like XP Minesweeper
                if i == self.triggered_mine:
                    grid.append('Mine Triggered')
                    continue
                elif i in self.mines and i not in self.flagged:
                    grid.append('Mine')
                    continue
                elif i in self.flagged and i not in self.mines:
                    grid.append('Mine X')
                    continue

            if i in self.flagged:
                grid.append('Flagged')
            elif i in self.revealed:
                count = self._adjacent_mine_count(i)
                grid.append('Checked' if count == 0 else str(count))
            else:
                grid.append('Unchecked')

        return grid

# Trains the DQN on forced-guess states while letting the CSP handle
# all deterministic moves, matching how main.py runs in production.
def train_hybrid_agent(
    episodes=10000,
    rows=TILE_ROWS,
    cols=TILE_COLUMNS,
    num_mines=TOTAL_MINES,
    save_path=MODEL_PATH,
):
    save_path = resolve_model_path(save_path)

    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
    agent = DQNAgent(
        rows=rows,
        cols=cols,
        total_mines=num_mines,
        lr=1e-3,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
    )

    starting_tile = (rows // 2) * cols + (cols // 2)
    wins = 0
    recent_wins = deque(maxlen=500)

    for ep in range(1, episodes + 1):
        state = env.reset()

        # First move: pick center tile to maximize flood-fill opening
        state, _, done, info = env.step(starting_tile)

        while not done:
            # 1. Let CSP solve all deterministic flags & picks first
            safe_flags, safe_picks, invalid_flags = analyze_grid_with_csp(
                state, total_mines=num_mines, rows=rows, cols=cols
            )

            if safe_flags or safe_picks or invalid_flags:
                for f in safe_flags:
                    env.flag(f)
                for p in safe_picks:
                    state, _, done, info = env.step(p)
                    if done:
                        break
                if not done:
                    state = env.get_grid_state()
                continue

            # 2. CSP found no guaranteed moves -> Ask DQN to make an educated guess
            action = agent.select_action(state, training=True)
            if action is None:
                break

            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state

        won = int(info.get("won", False))
        wins += won
        recent_wins.append(won)

        if ep % 500 == 0:
            win_rate = (sum(recent_wins) / len(recent_wins)) * 100
            print(
                f"Episode {ep}/{episodes} | "
                f"Recent Win Rate: {win_rate:.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            log_file = Path(f"{TRAINING_FOLDER}/dqn_training_log.csv")
            if ep == 500:
                log_file.write_text("episode,win_rate,epsilon\n")
            with log_file.open("a") as f:
                f.write(f"{ep},{win_rate:.2f},{agent.epsilon:.4f}\n")


    agent.save(save_path)
    print(f"\nTraining complete! Saved weights to {save_path}")

# Evaluates the hybrid agent (CSP + DQN) on a specified number of games.
def evaluate_hybrid_agent(
    games=1000,
    rows=TILE_ROWS,
    cols=TILE_COLUMNS,
    num_mines=TOTAL_MINES,
    model_path=MODEL_PATH,
    supervised=False,
):
    model_path = resolve_model_path(model_path)

    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
    agent = DQNAgent(rows=rows, cols=cols, total_mines=num_mines)
    agent.load(str(model_path))

    starting_tile = (rows // 2) * cols + (cols // 2)
    wins = 0

    # play the specified number of games and track wins.
    for _ in range(games):
        state = env.reset()
        state, _, done, info = env.step(starting_tile)

        while not done:
            safe_flags, safe_picks, _ = analyze_grid_with_csp(
                state, total_mines=num_mines, rows=rows, cols=cols
            )
            if safe_flags or safe_picks:
                for f in safe_flags:
                    env.flag(f)
                for p in safe_picks:
                    state, _, done, info = env.step(p)
                    if done:
                        break
                if not done:
                    state = env.get_grid_state()
                continue

            action = agent.select_safest_action(state) if supervised else agent.select_action(state, training=False)
            if action is None:
                break
            state, _, done, info = env.step(action)

        wins += int(info.get("won", False))

    win_rate = (wins / games) * 100
    print(f"Evaluated {games} games on {rows}x{cols} ({num_mines} mines): {win_rate:.2f}% win rate ({wins}/{games})")

def train_supervised_agent(
    episodes=20000,
    rows=TILE_ROWS,
    cols=TILE_COLUMNS,
    num_mines=TOTAL_MINES,
    save_path="minesweeper_sl_9x9_10m.pth",
    batch_size=128,
):
    save_path = resolve_model_path(save_path)
    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
    agent = DQNAgent(rows=rows, cols=cols, total_mines=num_mines, lr=1e-3)

    starting_tile = (rows // 2) * cols + (cols // 2)
    state_buffer = []
    mine_buffer = []
    recent_wins = deque(maxlen=500)
    total_loss = 0.0
    loss_steps = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        state, _, done, info = env.step(starting_tile)

        while not done:
            safe_flags, safe_picks, invalid_flags = analyze_grid_with_csp(
                state, total_mines=num_mines, rows=rows, cols=cols
            )
            if safe_flags or safe_picks or invalid_flags:
                for f in safe_flags:
                    env.flag(f)
                for p in safe_picks:
                    state, _, done, info = env.step(p)
                    if done:
                        break
                if not done:
                    state = env.get_grid_state()
                continue

            # Record this forced-guess board state and its exact hidden mine map
            state_buffer.append(list(state))
            mine_buffer.append(set(env.mines))

            # Early episodes explore randomly to diversify board states; later episodes use model
            if ep < episodes // 4 or random.random() < 0.15:
                valid = agent.get_valid_actions(state)
                action = random.choice(valid) if valid else None
            else:
                action = agent.select_safest_action(state)

            if action is None:
                break

            state, _, done, info = env.step(action)

            # Train whenever we have collected a full mini-batch
            if len(state_buffer) >= batch_size:
                loss = agent.train_supervised_batch(state_buffer, mine_buffer)
                total_loss += loss
                loss_steps += 1
                state_buffer.clear()
                mine_buffer.clear()

        recent_wins.append(int(info.get("won", False)))

        if ep % 1000 == 0:
            win_rate = (sum(recent_wins) / len(recent_wins)) * 100
            avg_loss = total_loss / max(1, loss_steps)
            print(
                f"Episode {ep}/{episodes} | "
                f"Recent Win Rate: {win_rate:.1f}% | "
                f"BCE Loss: {avg_loss:.4f}"
            )
            total_loss = 0.0
            loss_steps = 0
            log_file = Path(f"{TRAINING_FOLDER}/sl_training_log.csv")
            if ep == 1000:
                log_file.write_text("episode,win_rate,bce_loss\n")
            with log_file.open("a") as f:
                f.write(f"{ep},{win_rate:.2f},{avg_loss:.4f}\n")

    agent.save(save_path)
    print(f"\nSupervised training complete! Saved weights to {save_path}")


if __name__ == "__main__":
    import csp
    csp.DEBUG_CSP = False

    mode = "train"
    episodes = 10000
    rows = TILE_ROWS
    cols = TILE_COLUMNS
    num_mines = TOTAL_MINES
    model_path = MODEL_PATH

    for arg in sys.argv[1:]:
        lower_arg = arg.lower()
        if lower_arg in ("train", "eval", "train_sl", "eval_sl"):
            mode = lower_arg
        elif lower_arg.startswith("episodes=") or lower_arg.startswith("games="):
            episodes = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("rows="):
            rows = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("cols="):
            cols = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("mines="):
            num_mines = int(lower_arg.split("=", 1)[1])
        elif lower_arg.startswith("model=") or lower_arg.startswith("save_path="):
            raw_path = Path(arg.split("=", 1)[1])
            model_path = MODELS_FOLDER / raw_path.name if raw_path.parent == Path(".") else raw_path
        else:
            raise ValueError(f"Unknown argument: {arg!r}")

    if mode == "eval":
        evaluate_hybrid_agent(games=episodes, rows=rows, cols=cols, num_mines=num_mines, model_path=model_path)
    elif mode == "eval_sl":
        evaluate_hybrid_agent(games=episodes, rows=rows, cols=cols, num_mines=num_mines, model_path=model_path, supervised=True)
    elif mode == "train_sl":
        train_supervised_agent(episodes=episodes, rows=rows, cols=cols, num_mines=num_mines, save_path=model_path)
    else:
        train_hybrid_agent(episodes=episodes, rows=rows, cols=cols, num_mines=num_mines, save_path=model_path)
