import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import TILE_ROWS, TILE_COLUMNS

# Map string tile classes into numerical channels/values for the neural network
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

GAMMA = 0.1

# Encodes the Minesweeper grid into a 2-channel 2D array for the neural network.
# Input: grid_state (list of all tile strings)
# Output: 2-channel 2D array (2, TILE_ROWS, TILE_COLUMNS) suitable for DQN input
def encode_grid(grid_state):
    values = np.zeros((TILE_ROWS, TILE_COLUMNS), dtype=np.float32)
    clickable_mask = np.zeros((TILE_ROWS, TILE_COLUMNS), dtype=np.float32)

    for idx, label in enumerate(grid_state):
        r = idx // TILE_COLUMNS
        c = idx % TILE_COLUMNS
        values[r, c] = STATE_ENCODING.get(label, -1.0)
        if label in ('Unchecked', 'Unchecked Question'):
            clickable_mask[r, c] = 1.0

    stacked = np.stack([values, clickable_mask], axis=0)  # Shape: (2, 9, 9)
    return stacked


class MinesweeperCNN(nn.Module):
    # Convolutional Q-Network that outputs a Q-value for every tile (81 actions).
    def __init__(self, rows=TILE_ROWS, cols=TILE_COLUMNS):
        super().__init__()
        self.num_actions = rows * cols

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * rows * cols, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.head(features)

# Deep Q-Network agent for Minesweeper. Handles action selection, experience replay, and training.
class DQNAgent:
    # Initializes the DQN agent with hyperparameters, replay buffer, and neural networks.
    # Do I understand all this math stuff? Not necessarily,
    # but I know how to implement it and that's what counts.
    def __init__(
        self,
        lr=1e-3,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=200,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = TILE_ROWS * TILE_COLUMNS
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        self.policy_net = MinesweeperCNN().to(self.device)
        self.target_net = MinesweeperCNN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for training stability
        self.memory = deque(maxlen=buffer_size)

    # Returns indices of tiles that are still unrevealed and unflagged.
    def get_valid_actions(self, grid_state):
        return [
            i for i, label in enumerate(grid_state)
            if label in ('Unchecked', 'Unchecked Question')
        ]

    # Selects the best tile index to click using epsilon-greedy exploration
    # and strict action masking over valid 'Unchecked' tiles.
    def select_action(self, grid_state, training=True):
        # Get the list of valid actions (tiles that are still 'Unchecked' or 'Unchecked Question')
        valid_actions = self.get_valid_actions(grid_state)
        if not valid_actions:
            return None

        # Epsilon-greedy random exploration (only among valid tiles)
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.tensor(encode_grid(grid_state), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)  # Shape: (81,)

        # Mask out invalid tiles (already revealed/flagged) by setting Q-value to -infinity
        mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
        mask[valid_actions] = 0.0
        masked_q_values = q_values + mask

        return int(torch.argmax(masked_q_values).item())

    # Stores an experience tuple in the replay buffer.
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((
            encode_grid(state),
            action,
            reward,
            encode_grid(next_state),
            done
        ))

    # Samples a mini-batch from replay memory and updates the policy network.
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q = r + gamma * max_a' Q_target(s', a') (masked to valid next moves)
        with torch.no_grad():
            next_q_raw = self.target_net(next_states_t)
            # Channel 1 of next_states_t is the valid clickable mask (1.0 = valid, 0.0 = invalid)
            valid_next_mask = next_states_t[:, 1, :, :].view(self.batch_size, -1) > 0.5
            next_q_raw[~valid_next_mask] = float('-inf')
            max_next_q = next_q_raw.max(dim=1)[0]
            # Handle terminal states where no valid moves remain
            max_next_q = torch.where(torch.isinf(max_next_q), torch.zeros_like(max_next_q), max_next_q)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate and sync target network periodically
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path="minesweeper_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="minesweeper_dqn.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
