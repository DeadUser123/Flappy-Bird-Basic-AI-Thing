# Kevin Xia — Basic Flappy Bird AI using DDQN stuff
# got to like 230 ish before I quit because it was boring
# does random stuff during training so scores are lower than during playback
# uses prioritized experience replay because normal experience replay was too slow

import random
import math
import time
import csv
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os

# le parameters
SEED = 42 # the answer to life, the universe, and everything
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 200000
INITIAL_REPLAY_SIZE = 2000
TRAIN_EVERY = 4
TARGET_TAU = 0.005
PRIOR_ALPHA = 0.6
PRIOR_BETA_START = 0.4
PRIOR_BETA_FRAMES = 200000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99995
MAX_GRAD_NORM = 1.0
SAVE_DIR = "checkpoints"
LOG_CSV = "training_log.csv"

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class FlappyBirdEnv: # simulate flappy bird (was too lazy to find one to pull off from internet)
    def __init__(self, episode=0):
        self.gravity = 0.5
        self.flap_strength = -8.0
        self.bird_x = 50.0
        self.bird_y = 250.0
        self.bird_vel = 0.0
        self.pipe_width = 50
        self.screen_width = 400
        self.screen_height = 500
        self.pipes = []
        self.score = 0
        self.bird_radius = 12

        # curriculum to make things easier early on (because the thing kept dying a lot)
        self.episode = episode
        self.easy_gap = 200
        self.normal_gap = 120
        self.pipe_gap = self.easy_gap if episode < 1000 else self.normal_gap

        # state: normalized features (bird_y, bird_vel, x_dist_to_pipe, y_diff_to_gap_center, pipe_y)
        self.state_size = 5
        self.action_size = 2

        self.reset()

    def reset(self):
        self.bird_y = 250.0
        self.bird_vel = 0.0
        gap_margin = int(self.pipe_gap // 2 + 20)
        self.pipes = [[200.0, float(random.randint(gap_margin, self.screen_height - gap_margin))]]
        self.score = 0
        self._passed_current = False
        return self.get_state()

    def get_state(self):
        pipe_x, pipe_y = self.pipes[0]
        dx = (pipe_x - self.bird_x) / float(self.screen_width) # distance between bird and pipe
        dy = (self.bird_y - pipe_y) / float(self.screen_height) # vertical distance between bird and pipe center
        vy = np.clip(self.bird_vel / 10.0, -1.0, 1.0) # bird vertical velocity
        by = np.clip(self.bird_y / float(self.screen_height), 0.0, 1.0) # bird y position
        py = np.clip(pipe_y / float(self.screen_height), 0.0, 1.0)  # pipe y position (forgor that dy existed)
        return np.array([by, vy, np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0), py], dtype=np.float32)

    def step(self, action): # basically next frame
        if action == 1:
            self.bird_vel = self.flap_strength
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        for p in self.pipes:
            p[0] -= 3.0

        reward = 0.01
        if action == 1:
            reward -= 0.01

        pipe_x, pipe_y = self.pipes[0]
        gap_top = pipe_y - self.pipe_gap / 2.0
        gap_bot = pipe_y + self.pipe_gap / 2.0
        done = False

        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius

        # Collision
        if (pipe_x < bird_right and pipe_x + self.pipe_width > bird_left):
            if not (gap_top < self.bird_y < gap_bot):
                done = True
                reward = -10.0

        if self.bird_y <= 0 or self.bird_y >= self.screen_height:
            done = True
            reward = -10.0

        # Enhanced shaping: reward based on distance to gap center to encourage being alive
        gap_center = pipe_y
        dist_to_center = abs(self.bird_y - gap_center)
        reward += max(0.0, (self.pipe_gap / 2 - dist_to_center) / (self.pipe_gap / 2)) * 0.1

        # Scoring
        if pipe_x + self.pipe_width < self.bird_x and not self._passed_current:
            self.score += 1
            reward += 1.0
            self._passed_current = True

        # Recycle pipe
        if pipe_x < -self.pipe_width:
            self.pipes.pop(0)
            gap_margin = int(self.pipe_gap // 2 + 20)
            self.pipes.append([float(self.screen_width), float(random.randint(gap_margin, self.screen_height - gap_margin))])
            self._passed_current = False
            self.pipe_gap = self.normal_gap

        reward = float(np.clip(reward, -20.0, 20.0))
        return self.get_state(), reward, done, {}

class DQNNet(nn.Module): # normal dqn
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplay: # because it wasn't learning fast enough
    def __init__(self, capacity, alpha=0.6, seed=SEED):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.full = False
        self.rng = np.random.RandomState(seed)
        self.states = np.zeros((capacity, 5), dtype=np.float32)  # updated to 5
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, 5), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, state, action, reward, next_state, done):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.priorities[idx] = self.max_priority
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def sample(self, batch_size, beta=0.4):
        size = len(self)
        prios = self.priorities[:size] ** self.alpha
        probs = prios / prios.sum()
        indices = self.rng.choice(size, batch_size, p=probs)
        total = size
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        batch = dict(
            states = torch.tensor(self.states[indices], dtype=torch.float32, device=DEVICE),
            actions = torch.tensor(self.actions[indices], dtype=torch.long, device=DEVICE),
            rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=DEVICE),
            next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=DEVICE),
            dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=DEVICE),
            indices = indices,
            weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        )
        return batch

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        td = np.abs(td_errors) + epsilon
        for idx, val in zip(indices, td):
            self.priorities[idx] = val
            if val > self.max_priority:
                self.max_priority = val

# double dqn agent because single dqn kept dying
class Agent:
    def __init__(self, state_dim, action_dim):
        self.q = DQNNet(state_dim, action_dim).to(DEVICE)
        self.target = DQNNet(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.replay = PrioritizedReplay(BUFFER_SIZE, alpha=PRIOR_ALPHA)
        self.step_count = 0
        self.beta = PRIOR_BETA_START
        self.epsilon = EPSILON_START

    def act(self, state, eval_mode=False):
        if (not eval_mode) and (random.random() < self.epsilon):
            return random.randrange(2)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(state_t)
            return int(torch.argmax(qvals, dim=1).item())

    def soft_update_target(self, tau=TARGET_TAU):
        for param, target_param in zip(self.q.parameters(), self.target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def store(self, s,a,r,ns,d):
        self.replay.add(s,a,r,ns,d)

    def train_step(self):
        if len(self.replay) < INITIAL_REPLAY_SIZE:
            return None
        self.step_count += 1
        frac = min(1.0, self.step_count / float(PRIOR_BETA_FRAMES))
        self.beta = PRIOR_BETA_START + frac * (1.0 - PRIOR_BETA_START)
        batch = self.replay.sample(BATCH_SIZE, beta=self.beta)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        weights = batch['weights']
        indices = batch['indices']

        with torch.no_grad():
            next_q_online = self.q(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target(next_states)
            next_q_value = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * GAMMA * next_q_value

        q_values = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = (target_q - q_values).detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)
        loss = F.smooth_l1_loss(q_values * weights, target_q * weights)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), MAX_GRAD_NORM)
        self.opt.step()
        self.soft_update_target()
        return loss.item()

def save_checkpoint(agent, episode, path=SAVE_DIR): # saves the agent so can use again
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"dqn_ep{episode}.pth")
    torch.save({
        'q_state_dict': agent.q.state_dict(),
        'target_state_dict': agent.target.state_dict(),
        'opt_state_dict': agent.opt.state_dict(),
        'step_count': agent.step_count,
        'epsilon': agent.epsilon
    }, fname)
    print(f"Saved checkpoint: {fname}")

def train(episodes=2000, checkpoint_every=500): # 2000 was an arbitrary choice, 1500 is probably enough haven't tried though
    env = FlappyBirdEnv()
    agent = Agent(env.state_size, env.action_size)
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode','reward','score','steps','epsilon','loss'])

    total_steps = 0
    for ep in range(1, episodes+1):
        env = FlappyBirdEnv(ep)
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        losses = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if total_steps % TRAIN_EVERY == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if agent.epsilon > EPSILON_END:
                agent.epsilon *= EPSILON_DECAY
                agent.epsilon = max(agent.epsilon, EPSILON_END)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, round(ep_reward,4), env.score, ep_steps, round(agent.epsilon,4), round(avg_loss,6)])
        print(f"Episode {ep}: Reward = {ep_reward:.2f}, Eps = {agent.epsilon:.4f}, Score = {env.score}, Steps = {ep_steps}, AvgLoss = {avg_loss:.4f}") # score might not be reliable for some reason idk why it's like that

        if ep % checkpoint_every == 0:
            save_checkpoint(agent, ep)

    save_checkpoint(agent, episodes)
    return agent

# runs the game using the trained agent
def play_game(agent, speed=30, episodes=5):
    pygame.init()
    screen = pygame.display.set_mode((400, 500))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    for ep in range(episodes):
        env = FlappyBirdEnv()
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state, eval_mode=True) # no more random actions = higher scores than training
            state, _, done, _ = env.step(action)
            screen.fill((135, 206, 235))
            for (pipe_x, pipe_y) in env.pipes:
                pygame.draw.rect(screen, (0, 180, 0), (pipe_x, 0, env.pipe_width, pipe_y - env.pipe_gap//2))
                pygame.draw.rect(screen, (0, 180, 0), (pipe_x, pipe_y + env.pipe_gap//2, env.pipe_width, env.screen_height - (pipe_y + env.pipe_gap//2)))
            pygame.draw.circle(screen, (255, 255, 0), (int(env.bird_x), int(env.bird_y)), env.bird_radius)
            score_text = font.render(f"Score: {env.score}", True, (0,0,0))
            screen.blit(score_text, (10,10))
            pygame.display.flip()
            clock.tick(speed)
    pygame.quit()

# actually run it for real
agent = train(episodes=2000, checkpoint_every=500)
play_game(agent, speed=30, episodes=5)
