import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import TILE_ROWS, TILE_COLUMNS

DEBUG_DQN = False

# Translate string tile labels into normalized numbers so the CNN can process them.
# Numbers 1-8 are scaled between 0.125 and 1.0, while special states get negative values.
STATE_ENCODING = {
    'Unchecked': -1.0,
    'Unchecked Question': -1.0,
    'Flagged': -2.0,
    'Checked': 0.0,
    '1': 1.0 / 8.0,
    '2': 2.0 / 8.0,
    '3': 3.0 / 8.0,
    '4': 4.0 / 8.0,
    '5': 5.0 / 8.0,
    '6': 6.0 / 8.0,
    '7': 7.0 / 8.0,
    '8': 8.0 / 8.0,
    'Mine': -3.0,
    'Mine Triggered': -3.0,
    'Mine X': -3.0,
}

# Discount factor (0.0 to 1.0): controls how much the agent values future rewards
# vs. immediate survival. Kept low (0.1) because Minesweeper guesses need immediate safety.
GAMMA = 0.1


# Encodes the Minesweeper grid into a 3-channel 2D array (3, rows, cols).
# Channel 0 tells the network what each tile is (numbers 1–8, flagged, or unchecked)
# Channel 1 acts as a binary map (1 or 0) telling it which tiles are legal to click.
# Channel 2 is the remaining mine ratio (remaining_mines / remaining_unchecked), 
# giving the local CNN global risk context.
def encode_grid(grid_state, rows=TILE_ROWS, cols=TILE_COLUMNS, total_mines=TOTAL_MINES):
    values = np.zeros((rows, cols), dtype=np.float32)
    clickable_mask = np.zeros((rows, cols), dtype=np.float32)

    flagged_count = 0
    unchecked_count = 0

    for idx, label in enumerate(grid_state):
        r = idx // cols
        c = idx % cols
        values[r, c] = STATE_ENCODING.get(label, -1.0)
        if label in ('Unchecked', 'Unchecked Question'):
            clickable_mask[r, c] = 1.0
            unchecked_count += 1
        elif label == 'Flagged':
            flagged_count += 1

    remaining_mines = max(0, total_mines - flagged_count)
    density = remaining_mines / max(1, unchecked_count)
    mine_density_channel = np.full((rows, cols), density, dtype=np.float32)

    return np.stack([values, clickable_mask, mine_density_channel], axis=0)

# Fully Convolutional Q-Network:
# Takes a (batch, 3, rows, cols) board tensor and predicts a Q-value (expected reward)
# for every tile on the board. Because it uses only convolutional layers (no Linear layers),
# the same trained weights work on any board size (6x6, 9x9, 16x30, etc.).
class MinesweeperCNN(nn.Module):
    # Fully convolutional Q-Network: now it works on any (rows, cols) board size, yippee!!
    def __init__(self, rows=TILE_ROWS, cols=TILE_COLUMNS):
        super().__init__()
        # Three 3x3 conv layers with padding=1 preserve the rows and columns
        # while letting each tile "see" a 7x7 neighborhood around itself.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 1x1 convolutions compress the 64 feature channels at each tile down to 
        # a single Q-value per tile, then flatten it into a 1D list of length (rows * cols).
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, x):
        features = self.conv(x)
        return self.head(features)

# Deep Q-Network (DQN) Agent
# Learns which unrevealed tiles are safest to guess when the deterministic CSP solver gets stuck.
class DQNAgent:
    def __init__(
        self,
        rows=TILE_ROWS,
        cols=TILE_COLUMNS,
        lr=1e-3,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=200,
        total_mines=TOTAL_MINES,
    ):
        # use GPU if available, otherwise fall back to CPU
        # if this were run on a better laptop than mine this would be useful lol 
        # (added it anyway cuz i might run this on my home computer later)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rows = rows
        self.cols = cols
        self.num_actions = rows * cols
        self.gamma = gamma

        # Exploration vs. exploitation parameters (epsilon decays from 100% random down to 5%)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Experience replay & target sync settings
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        # policy_net is actively trained, target_net gives stable Q-value targets during updates
        self.policy_net = MinesweeperCNN(rows=rows, cols=cols).to(self.device)
        self.target_net = MinesweeperCNN(rows=rows, cols=cols).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly via backprop

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()    # Huber loss prevents exploding gradients from large penalties
        self.memory = deque(maxlen=buffer_size) # circular replay buffer storing past transitions
        self.total_mines = total_mines

    # return a list of 1d  tile indices that aren't revealed or flagged
    def get_valid_actions(self, grid_state):
        # Frontier Masking:
        #   1. Finds all unrevealed tiles.
        #   2. Filters them down to 'frontier' tiles — unrevealed tiles touching at least
        #      one revealed number, which maximize information gain for the CSP.
        #   3. Falls back to all unrevealed tiles if no frontier tiles exist (e.g., sealed pocket).
        all_unchecked = [
            i for i, label in enumerate(grid_state)
            if label in ('Unchecked', 'Unchecked Question')
        ]
        if not all_unchecked:
            return []

        revealed_numbers = {'1', '2', '3', '4', '5', '6', '7', '8'}
        frontier = []

        for idx in all_unchecked:
            r, c = divmod(idx, self.cols)
            has_number_neighbor = False

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if grid_state[nr * self.cols + nc] in revealed_numbers:
                            has_number_neighbor = True
                            break
                if has_number_neighbor:
                    break

            if has_number_neighbor:
                frontier.append(idx)

        # Return frontier tiles if available; otherwise fall back to any unrevealed tile
        return frontier if frontier else all_unchecked


    # chooses a tile index to click:
    # during training, picks randomly with probability epsilon to explore new states.
    # otherwise it just defaults to the CNN and picks the valid unrevealed tile with the highest q-value
    def select_action(self, grid_state, training=True):
        valid_actions = self.get_valid_actions(grid_state)
        if not valid_actions:
            return None

        # epsilon-greedy exploration (valid tiles ONLY!!)
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Convert board to tensor and add batch dimension: (1, 2, rows, cols)
        encoded = encode_grid(grid_state, rows=self.rows, cols=self.cols, total_mines=self.total_mines)
        state_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():   # Disable gradient tracking during inference for speed
            q_values = self.policy_net(state_tensor).squeeze(0) # shape: rows * cols

        # fill invalid tiles with -infinity so argmax can never select them
        mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
        mask[valid_actions] = 0.0
        masked_q_values = q_values + mask

        return int(torch.argmax(masked_q_values).item())

    # Supervised Inference:
    # Predicts the mine probability (0.0 to 1.0) for every tile using sigmoid(logits),
    # then selects the valid frontier tile with the LOWEST predicted mine probability.
    def select_safest_action(self, grid_state):
        valid_actions = self.get_valid_actions(grid_state)
        if not valid_actions:
            return None

        encoded = encode_grid(grid_state, rows=self.rows, cols=self.cols, total_mines=self.total_mines)
        state_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy_net(state_tensor).squeeze(0)
            mine_probs = torch.sigmoid(logits)

        # Mask out invalid/non-frontier tiles with +infinity so argmin picks the safest valid tile
        mask = torch.full((self.num_actions,), float('inf'), device=self.device)
        mask[valid_actions] = 0.0
        masked_probs = mine_probs + mask

        return int(torch.argmin(masked_probs).item())

    # Trains policy_net using Binary Cross-Entropy (BCEWithLogitsLoss) against the
    # true hidden mine locations, masked strictly to unrevealed tiles.
    def train_supervised_batch(self, states_list, mine_sets):
        if not states_list:
            return 0.0

        encoded_states = np.array([
            encode_grid(s, rows=self.rows, cols=self.cols, total_mines=self.total_mines) for s in states_list
        ])
        states_t = torch.tensor(encoded_states, dtype=torch.float32, device=self.device)

        # Build ground-truth binary targets (1.0 = mine, 0.0 = safe) and unrevealed masks
        targets = np.zeros((len(states_list), self.num_actions), dtype=np.float32)
        unrevealed_mask = np.zeros((len(states_list), self.num_actions), dtype=np.float32)

        for row_idx, (state, mines) in enumerate(zip(states_list, mine_sets)):
            for tile_idx, label in enumerate(state):
                if label in ('Unchecked', 'Unchecked Question'):
                    unrevealed_mask[row_idx, tile_idx] = 1.0
                    if tile_idx in mines:
                        targets[row_idx, tile_idx] = 1.0

        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(unrevealed_mask, dtype=torch.float32, device=self.device)

        logits = self.policy_net(states_t)

        # Compute per-tile BCE loss and average only over unrevealed tiles
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_t, reduction='none')
        masked_loss = (bce * mask_t).sum() / mask_t.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        masked_loss.backward()
        self.optimizer.step()

        return masked_loss.item()


    # Saves a single (state, action, reward, next_state, done) experience into replay memory.
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((
            encode_grid(state, rows=self.rows, cols=self.cols, total_mines=self.total_mines),
            action,
            reward,
            encode_grid(next_state, rows=self.rows, cols=self.cols, total_mines=self.total_mines),
            done
        ))


    # Samples a random mini-batch from replay memory and performs one gradient descent step
    # using the Bellman equation: Target Q = reward + gamma * max(Q_target(next_state)).
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None # wait until enough experiences have been collected

        # Sample a batch of transitions from the replay memory for training
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert batch tuples into tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Predicted Q-value for the specific action the agent actually took in each state
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Compute Bellman target Q-values using the target network without tracking gradients
        with torch.no_grad():
            next_q_raw = self.target_net(next_states_t)
            # use channel 1 (clickable mask) to mask out invalid moves for next state
            valid_next_mask = next_states_t[:, 1, :, :].view(self.batch_size, -1) > 0.5
            next_q_raw[~valid_next_mask] = float('-inf')
            # best q value from next state
            max_next_q = next_q_raw.max(dim=1)[0]
            # if game ended, replace with 0
            max_next_q = torch.where(torch.isinf(max_next_q), torch.zeros_like(max_next_q), max_next_q)
            # if game is done, future reward = 0, otherwise add discounted future Q
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # Backpropagate loss and update policy_net weights
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon and periodically sync target network with policy network
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


    # these two are just for saving and loading the model weights
    def save(self, path="minesweeper_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="minesweeper_dqn.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
