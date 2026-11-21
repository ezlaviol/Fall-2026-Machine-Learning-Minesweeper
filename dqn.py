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


# Encodes the Minesweeper grid into a 2-channel 2D array (2, rows, cols).
# Channel 0 tells the network what each tile is (numbers 1–8, flagged, or unchecked)
# Channel 1 acts as a binary map (1 or 0) telling it which tiles are legal to click.
def encode_grid(grid_state, rows=TILE_ROWS, cols=TILE_COLUMNS):
    values = np.zeros((rows, cols), dtype=np.float32)
    clickable_mask = np.zeros((rows, cols), dtype=np.float32)

    for idx, label in enumerate(grid_state):
        r = idx // cols
        c = idx % cols
        values[r, c] = STATE_ENCODING.get(label, -1.0)
        if label in ('Unchecked', 'Unchecked Question'):
            clickable_mask[r, c] = 1.0

    # Stack both 2D arrays along axis 0 to create a 2-channel "image" for the CNN
    return np.stack([values, clickable_mask], axis=0)

# Fully Convolutional Q-Network:
# Takes a (batch, 2, rows, cols) board tensor and predicts a Q-value (expected reward)
# for every tile on the board. Because it uses only convolutional layers (no Linear layers),
# the same trained weights work on any board size (6x6, 9x9, 16x30, etc.).
class MinesweeperCNN(nn.Module):
    # Fully convolutional Q-Network: now it works on any (rows, cols) board size, yippee!!
    def __init__(self, rows=TILE_ROWS, cols=TILE_COLUMNS):
        super().__init__()
        # Three 3x3 conv layers with padding=1 preserve the rows and columns
        # while letting each tile "see" a 7x7 neighborhood around itself.
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
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
        encoded = encode_grid(grid_state, rows=self.rows, cols=self.cols)
        state_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():   # Disable gradient tracking during inference for speed
            q_values = self.policy_net(state_tensor).squeeze(0) # shape: rows * cols

        # fill invalid tiles with -infinity so argmax can never select them
        mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
        mask[valid_actions] = 0.0
        masked_q_values = q_values + mask

        return int(torch.argmax(masked_q_values).item())

    # Saves a single (state, action, reward, next_state, done) experience into replay memory.
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((
            encode_grid(state, rows=self.rows, cols=self.cols),
            action,
            reward,
            encode_grid(next_state, rows=self.rows, cols=self.cols),
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
