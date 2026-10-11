"""dqn_agent.py - Deep Q-Network e-puck for Webots (numpy only).

State  (11 floats): 8 normalised IR readings, normalised goal distance,
                    sin/cos of the bearing to the goal.
Actions (3):        FORWARD, LEFT, RIGHT.

Files written next to this controller:
  <name>.npz                    network weights (auto-loaded at startup)
  progress_<name>.csv           one line per episode (for plot_progress.py)

Delete the .npz file to learn from scratch. Add "--eval" to controllerArgs
for a greedy demonstration run after training. Set DOUBLE_DQN=False for a
vanilla-DQN comparison.
"""

import argparse
import csv
import math
import os
import random

import numpy as np
from controller import Supervisor

# ---------------------------------------------------------------- tuning --
GAMMA = 0.95            # discount factor
LR = 5e-4               # SGD learning rate
BATCH_SIZE = 64
BUFFER_SIZE = 30000     # experience replay capacity
WARMUP_STEPS = 500      # random experience collected before training starts
TRAIN_EVERY = 4         # one gradient step every N environment steps
TARGET_SYNC = 500       # env steps between target-network copies
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.95
HIDDEN = (64, 64)       # hidden layer sizes
GRAD_CLIP = 1.0
DOUBLE_DQN = True       # False = vanilla DQN (good comparison experiment!)

MAX_STEPS = 800
GOAL_RADIUS = 0.15
SHAPING = 5.0
STEP_PENALTY = 0.01
BUMP_PENALTY = 0.3
BUMP_THRESHOLD = 1500
MAX_SPEED = 6.28
TURN_SLOWDOWN = 0.2


class MLP:
    """From-scratch MLP: ReLU hidden layers, linear output (the Q-values)."""

    def __init__(self, layer_sizes, rng):
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            std = math.sqrt(2.0 / fan_in)                     # He init
            self.weights.append(
                rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, x):
        """Batch forward pass; keeps activations for backprop."""
        activations = [x]
        h = x
        for i in range(len(self.weights) - 1):
            h = np.maximum(h @ self.weights[i] + self.biases[i], 0.0)
            activations.append(h)
        return h @ self.weights[-1] + self.biases[-1], activations

    def q_values(self, state):
        q, _ = self.forward(state[None, :])
        return q[0]

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def train_step(self, states, actions, td_targets, lr):
        """One SGD step on the squared TD error of the taken actions."""
        q, acts = self.forward(states)
        rows = np.arange(len(actions))
        errors = q[rows, actions] - td_targets
        loss = float(np.mean(errors ** 2))

        grad = np.zeros_like(q)
        grad[rows, actions] = 2.0 * errors / len(actions)
        grad = np.clip(grad, -GRAD_CLIP, GRAD_CLIP)

        delta = grad
        for i in reversed(range(len(self.weights))):          # backprop
            gW = acts[i].T @ delta
            gB = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (acts[i] > 0.0)
            self.weights[i] -= lr * gW
            self.biases[i] -= lr * gB
        return loss


class ReplayBuffer:
    def __init__(self, capacity, state_dim, rng):
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.rng = capacity, rng
        self.index, self.size = 0, 0

    def add(self, s, a, r, s2, done):
        i = self.index
        self.states[i], self.next_states[i] = s, s2
        self.actions[i], self.rewards[i], self.dones[i] = a, r, done
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = self.rng.choice(self.size, size=n, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


class DqnEpuck(Supervisor):
    def __init__(self):
        super().__init__()
        p = argparse.ArgumentParser()
        p.add_argument("--goal", default="1.2,1.2", help="goal x,z in metres")
        p.add_argument("--dqn", default="dqn.npz", help="weights file")
        p.add_argument("--eval", action="store_true",
                       help="greedy demo mode: no exploration, no learning")
        p.add_argument("--seed", type=int, default=None)
        self.args = p.parse_args()
        self.goal = tuple(float(v) for v in self.args.goal.split(","))
        self.eval_mode = self.args.eval
        self.rng = random.Random(self.args.seed)
        self.nprng = np.random.default_rng(self.args.seed)

        folder = os.path.dirname(os.path.abspath(__file__))
        self.weights_file = os.path.join(folder, self.args.dqn)
        self.log_file = os.path.join(
            folder, "progress_" + os.path.splitext(self.args.dqn)[0] + ".csv")

        self.control_ts = 4 * int(self.getBasicTimeStep())
        self.ds = [self.getDevice("ps%d" % i) for i in range(8)]
        for s in self.ds:
            s.enable(self.control_ts)
        self.motors = [self.getDevice("left wheel motor"),
                       self.getDevice("right wheel motor")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        self.me = self.getSelf()
        self.translation = self.me.getField("translation")
        self.rotation = self.me.getField("rotation")
        self.start_pos = self.translation.getSFVec3f()

        shape = (11,) + HIDDEN + (3,)
        self.online = MLP(shape, self.nprng)
        self.target = MLP(shape, self.nprng)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(BUFFER_SIZE, 11, self.nprng)
        self._load_weights()

        self.epsilon = 0.0 if self.eval_mode else EPS_START
        self.episode = 0
        self.env_steps = 0
        self._new_episode(random_start=False)

    # ---------------------------------------------------------- weights ---
    def _load_weights(self):
        if not os.path.exists(self.weights_file):
            print("%s: no weights found, learning from scratch"
                  % self.getName(), flush=True)
            return
        data = np.load(self.weights_file)
        for i in range(len(self.online.weights)):
            self.online.weights[i] = data["W%d" % i]
            self.online.biases[i] = data["b%d" % i]
        self.target.copy_from(self.online)
        print("%s: weights loaded from %s"
              % (self.getName(), self.args.dqn), flush=True)

    def _save_weights(self):
        payload = {}
        for i in range(len(self.online.weights)):
            payload["W%d" % i] = self.online.weights[i]
            payload["b%d" % i] = self.online.biases[i]
        np.savez(self.weights_file, **payload)

    # ----------------------------------------------------------- senses ---
    def _ir(self):
        return [s.getValue() for s in self.ds]

    def _state(self):
        ir = np.asarray(self._ir(), dtype=np.float32) / 4095.0
        pos = self.me.getPosition()
        m = self.me.getOrientation()
        heading = math.atan2(m[2], m[8])                 # forward = local +z
        dist = math.hypot(self.goal[0] - pos[0], self.goal[2] - pos[2])
        bearing = math.atan2(self.goal[0] - pos[0], self.goal[2] - pos[2])
        rel = (bearing - heading + math.pi) % (2.0 * math.pi) - math.pi
        extra = np.array([min(dist, 3.0) / 3.0,
                          math.sin(rel), math.cos(rel)], dtype=np.float32)
        return np.concatenate([ir, extra]).astype(np.float32)

    def _reward(self, previous_distance):
        pos = self.me.getPosition()
        dist = math.hypot(self.goal[0] - pos[0], self.goal[2] - pos[2])
        r = SHAPING * (previous_distance - dist) - STEP_PENALTY
        ir = self._ir()
        if max(ir[7], ir[0]) > BUMP_THRESHOLD:
            r -= BUMP_PENALTY
        done = dist < GOAL_RADIUS
        if done:
            r += GOAL_REWARD
        return r, done, dist

    # ---------------------------------------------------------- actions ---
    def _drive(self, action):
        left = right = MAX_SPEED
        if action == 1:                                  # LEFT
            left = MAX_SPEED * TURN_SLOWDOWN
        elif action == 2:                                # RIGHT
            right = MAX_SPEED * TURN_SLOWDOWN
        self.motors[0].setVelocity(left)
        self.motors[1].setVelocity(right)

    def _act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(3)
        return int(np.argmax(self.online.q_values(state)))

    def _train(self):
        self.env_steps += 1
        if (self.eval_mode or self.env_steps < WARMUP_STEPS
                or self.buffer.size < BATCH_SIZE
                or self.env_steps % TRAIN_EVERY):
            return
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        if DOUBLE_DQN:                                   # Double DQN update
            best_next = np.argmax(self.online.forward(s2)[0], axis=1)
            q_next = self.target.forward(s2)[0][np.arange(BATCH_SIZE), best_next]
        else:                                            # vanilla DQN update
            q_next = self.target.forward(s2)[0].max(axis=1)
        targets = (r + GAMMA * (1.0 - d) * q_next).astype(np.float32)
        self.online.train_step(s, a, targets, LR)
        if self.env_steps % TARGET_SYNC == 0:
            self.target.copy_from(self.online)

    # --------------------------------------------------------- episodes ---
    def _new_episode(self, random_start=True):
        x, y, z = self.start_pos
        self.translation.setSFVec3f([x, y, z])
        angle = self.rng.uniform(0, 2 * math.pi) if random_start else 0.0
        self.rotation.setSFRotation([0, 1, 0, angle])
        self.me.resetPhysics()
        for m in self.motors:
            m.setVelocity(0.0)
        pos = self.me.getPosition()
        self.prev_distance = math.hypot(self.goal[0] - pos[0],
                                        self.goal[2] - pos[2])
        self.steps = 0
        self.episode_reward = 0.0

    def _end_episode(self, reached):
        self.episode += 1
        self._log(reached)
        if not self.eval_mode:
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        print("%s: episode %d: %s after %d steps, reward=%.1f, eps=%.2f"
              % (self.getName(), self.episode,
                 "GOAL" if reached else "timeout", self.steps,
                 self.episode_reward, self.epsilon), flush=True)
        if not self.eval_mode and self.episode % 10 == 0:
            self._save_weights()
        self._new_episode()

    def _log(self, reached):
        header = not os.path.exists(self.log_file)
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(["episode", "steps", "reward", "epsilon", "reached"])
            w.writerow([self.episode, self.steps,
                        round(self.episode_reward, 2),
                        round(self.epsilon, 3), int(reached)])

    # ------------------------------------------------------------- loop ---
    def run(self):
        while self.step(self.control_ts) != -1:
            state = self._state()
            action = self._act(state)
            self._drive(action)
            if self.step(self.control_ts) == -1:
                break
            reward, done, dist = self._reward(self.prev_distance)
            next_state = self._state()
            self.buffer.add(state, action, reward, next_state, float(done))
            self.episode_reward += reward
            self.prev_distance = dist
            self.steps += 1
            self._train()
            if done or self.steps >= MAX_STEPS:
                self._end_episode(done)


if __name__ == "__main__":
    DqnEpuck().run()
