"""vision_agent.py - camera-based learning e-puck for Webots.

  --mode=blob     tabular Q-learning over a colour-blob state:
                  (front,left,right) IR bits + beacon-visible bit +
                  column sector (0-7) + blob-height level (0-3), 512 states.
                  Reward uses ONLY camera-derived progress (blob height
                  change), so the learned policy is nearly deployable.
  --mode=pixels   Double-DQN over a downsampled colour grid - end-to-end
                  from pixels, no hand-designed detector. Slower to converge;
                  reward uses the simulator-provided distance (like a game
                  engine). Train with Webots fast mode.

Calibrate colours first with worlds/vision_check.wbt + vision_probe.
Requires beacon-style goals (worlds/vision_world.wbt).
"""
import argparse
import csv
import json
import math
import os
import random

import numpy as np
from controller import Supervisor

NAMED_COLORS = {
    "yellow": (217, 178, 26),
    "red": (200, 40, 30),
    "cyan": (30, 180, 200),
    "green": (40, 180, 60),
    "magenta": (200, 40, 170),
}

# ------------------------------------------------------------- shared -----
MAX_SPEED = 6.28
TURN_SLOWDOWN = 0.2
OBSTACLE_THRESHOLD = 600
BUMP_THRESHOLD = 1500
GOAL_RADIUS = 0.15
STEP_PENALTY = 0.01
BUMP_PENALTY = 0.3
GOAL_REWARD = 10.0
MAX_STEPS = 800
SAVE_EVERY = 10

# ------------------------------------------------------- blob (tabular) ---
TB_ALPHA = 0.2
TB_GAMMA = 0.95
TB_EPS_START, TB_EPS_MIN, TB_EPS_DECAY = 0.5, 0.05, 0.97
VIS_SHAPING = 8.0        # reward per unit of normalised blob-height gain
SEARCH_PENALTY = 0.05    # extra cost while the beacon is out of view
SECTORS = 8
MIN_BLOB = 4             # minimum matched pixels to count as "visible"
H_LEVELS = (0.06, 0.15, 0.30)   # normalised height -> distance level 0..3

# ------------------------------------------------------ pixels (DQN) ------
GAMMA = 0.95
LR = 5e-4
BATCH_SIZE = 32
BUFFER_SIZE = 20000
WARMUP = 1000
TRAIN_EVERY = 4
TARGET_SYNC = 400
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.95
HIDDEN = (64, 32)
GRAD_CLIP = 1.0

ACTIONS = ["FORWARD", "LEFT", "RIGHT"]


class MLP:
    """Same from-scratch network as dqn_agent."""

    def __init__(self, layer_sizes, rng):
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            std = math.sqrt(2.0 / fan_in)
            self.weights.append(
                rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, x):
        acts = [x]
        h = x
        for i in range(len(self.weights) - 1):
            h = np.maximum(h @ self.weights[i] + self.biases[i], 0.0)
            acts.append(h)
        return h @ self.weights[-1] + self.biases[-1], acts

    def q_values(self, state):
        q, _ = self.forward(state[None, :])
        return q[0]

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def train_step(self, states, actions, td_targets, lr):
        q, acts = self.forward(states)
        rows = np.arange(len(actions))
        errors = q[rows, actions] - td_targets
        grad = np.zeros_like(q)
        grad[rows, actions] = 2.0 * errors / len(actions)
        grad = np.clip(grad, -GRAD_CLIP, GRAD_CLIP)
        delta = grad
        for i in reversed(range(len(self.weights))):
            gW = acts[i].T @ delta
            gB = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (acts[i] > 0.0)
            self.weights[i] -= lr * gW
            self.biases[i] -= lr * gB
        return float(np.mean(errors ** 2))


class ReplayBuffer:
    def __init__(self, capacity, state_dim, rng):
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.cap, self.rng, self.i, self.n = capacity, rng, 0, 0

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.s2[self.i] = s, s2
        self.a[self.i], self.r[self.i], self.d[self.i] = a, r, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, n):
        idx = self.rng.choice(self.n, size=n, replace=False)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class VisionEpuck(Supervisor):
    def __init__(self):
        super().__init__()
        p = argparse.ArgumentParser()
        p.add_argument("--goal", default="1.2,1.2", help="goal x,z (m)")
        p.add_argument("--mode", choices=["blob", "pixels"], default="blob")
        p.add_argument("--color", choices=sorted(NAMED_COLORS),
                       default="yellow")
        p.add_argument("--rgb", default=None, help="custom goal colour R,G,B")
        p.add_argument("--tol", type=int, default=60)
        p.add_argument("--grid", default="12,8", help="cols,rows (pixels)")
        p.add_argument("--qtable", default="qtable_vision.json")
        p.add_argument("--weights", default="vision_dqn.npz")
        p.add_argument("--episodes", type=int, default=0,
                       help="greedy after N episodes (0 = never)")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--eval", action="store_true")
        self.args = p.parse_args()
        self.mode = self.args.mode
        if self.args.seed is not None:
            random.seed(self.args.seed)
        self.nprng = np.random.default_rng(self.args.seed)
        self.goal_rgb = (tuple(int(v) for v in self.args.rgb.split(","))
                         if self.args.rgb else NAMED_COLORS[self.args.color])
        self.goal = tuple(float(x) for x in self.args.goal.split(","))

        folder = os.path.dirname(os.path.abspath(__file__))
        self.qtable_file = os.path.join(folder, self.args.qtable)
        self.weights_file = os.path.join(folder, self.args.weights)
        self.log_file = os.path.join(folder,
                                     "progress_%s.csv" % self.getName())

        self.ts = 4 * int(self.getBasicTimeStep())
        self.cam = self.getDevice("camera")
        if self.cam is None:
            raise SystemExit(
                "no 'camera' device on this robot - use an E-puck PROTO "
                "that includes the camera (see README troubleshooting).")
        self.cam.enable(self.ts)
        self.cam_w, self.cam_h = self.cam.getWidth(), self.cam.getHeight()

        self.ds = [self.getDevice("ps%d" % i) for i in range(8)]
        for s in self.ds:
            s.enable(self.ts)
        self.motors = [self.getDevice("left wheel motor"),
                       self.getDevice("right wheel motor")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        self.me = self.getSelf()
        self.trans_field = self.me.getField("translation")
        self.rot_field = self.me.getField("rotation")
        self.start_pos = self.trans_field.getSFVec3f()

        self.env_steps = 0                       # global (never per-episode!)
        self.blob = None
        if self.mode == "blob":
            self.q = self._load_qtable()
            self.eps = 0.0 if self.args.eval else TB_EPS_START
        else:
            self.grid_c, self.grid_r = (int(v)
                                        for v in self.args.grid.split(","))
            self.in_dim = self.grid_c * self.grid_r * 3
            shape = (self.in_dim,) + HIDDEN + (3,)
            self.online = MLP(shape, self.nprng)
            self.target = MLP(shape, self.nprng)
            self.target.copy_from(self.online)
            self.buffer = ReplayBuffer(BUFFER_SIZE, self.in_dim, self.nprng)
            self._load_weights()
            self.eps = 0.0 if self.args.eval else EPS_START

        self.episode = 0
        self._new_episode(random_start=False)
        print("%s: mode=%s camera=%dx%d colour=%s tol=%d"
              % (self.getName(), self.mode, self.cam_w, self.cam_h,
                 self.goal_rgb, self.args.tol), flush=True)

    # ------------------------------------------------------ persistence ---
    def _load_qtable(self):
        try:
            with open(self.qtable_file) as f:
                return {int(k): [float(x) for x in v]
                        for k, v in json.load(f).items()}
        except Exception:
            return {}

    def _save_qtable(self):
        with open(self.qtable_file, "w") as f:
            json.dump(self.q, f)

    def _load_weights(self):
        if not os.path.exists(self.weights_file):
            return
        data = np.load(self.weights_file)
        for i in range(len(self.online.weights)):
            self.online.weights[i] = data["W%d" % i]
            self.online.biases[i] = data["b%d" % i]
        self.target.copy_from(self.online)
        print("%s: weights loaded" % self.getName(), flush=True)

    def _save_weights(self):
        payload = {}
        for i in range(len(self.online.weights)):
            payload["W%d" % i] = self.online.weights[i]
            payload["b%d" % i] = self.online.biases[i]
        np.savez(self.weights_file, **payload)

    # ------------------------------------------------------------ vision --
    def _scan(self):
        """Full-image colour match (vectorised). Camera image is BGRA."""
        img = self.cam.getImage()
        buf = np.frombuffer(img, dtype=np.uint8).astype(np.int16)
        buf = buf.reshape(self.cam_h, self.cam_w, 4)
        r, g, b = buf[:, :, 2], buf[:, :, 1], buf[:, :, 0]
        tr, tg, tb = self.goal_rgb
        tol = self.args.tol
        mask = ((np.abs(r - tr) < tol) & (np.abs(g - tg) < tol)
                & (np.abs(b - tb) < tol))
        ys, xs = np.nonzero(mask)
        self.blob = None
        if ys.size >= MIN_BLOB:
            self.blob = (float(xs.mean()) / self.cam_w,
                         float(ys.max() - ys.min() + 1), int(ys.size))

    def _blob_feats(self):
        if self.blob is None:
            return False, 0, 0.0, 0
        x_norm, h_px, _ = self.blob
        h_norm = h_px / self.cam_h
        sector = min(SECTORS - 1, int(x_norm * SECTORS))
        level = sum(h_norm > t for t in H_LEVELS)
        return True, sector, h_norm, level

    # ------------------------------------------------------------ states --
    def state_blob(self):
        v = [s.getValue() for s in self.ds]
        front = int(max(v[7], v[0]) > OBSTACLE_THRESHOLD)
        left = int(max(v[6], v[5]) > OBSTACLE_THRESHOLD)
        right = int(max(v[1], v[2]) > OBSTACLE_THRESHOLD)
        vis, sector, _, level = self._blob_feats()
        return (front | (left << 1) | (right << 2) | (vis << 3)
                | (sector << 4) | (level << 7))          # 512 states

    def state_pixels(self):
        buf = np.frombuffer(self.cam.getImage(),
                            dtype=np.uint8).astype(np.float32)
        buf = buf.reshape(self.cam_h, self.cam_w, 4)[:, :, :3] / 255.0
        ys = np.linspace(0, self.cam_h, self.grid_r + 1, dtype=int)
        xs = np.linspace(0, self.cam_w, self.grid_c + 1, dtype=int)
        cells = np.empty((self.grid_r, self.grid_c, 3), dtype=np.float32)
        for i in range(self.grid_r):
            for j in range(self.grid_c):
                cells[i, j] = buf[ys[i]:ys[i + 1],
                                  xs[j]:xs[j + 1]].mean(axis=(0, 1))
        return cells.reshape(-1)

    def goal_distance(self):
        pos = self.me.getPosition()
        return math.hypot(self.goal[0] - pos[0], self.goal[2] - pos[2])

    # ----------------------------------------------------------- actions --
    def drive(self, action):
        l = r = MAX_SPEED
        if action == 1:
            l = MAX_SPEED * TURN_SLOWDOWN
        elif action == 2:
            r = MAX_SPEED * TURN_SLOWDOWN
        self.motors[0].setVelocity(l)
        self.motors[1].setVelocity(r)

    def choose_tabular(self, state):
        if random.random() < self.eps:
            return random.randrange(3)
        row = self.q.setdefault(state, [0.0] * 3)
        best = max(row)
        return random.choice([a for a, v in enumerate(row) if v == best])

    def learn_tabular(self, state, action, r, next_state, done):
        target = r if done else r + TB_GAMMA * max(
            self.q.setdefault(next_state, [0.0] * 3))
        row = self.q.setdefault(state, [0.0] * 3)
        row[action] += TB_ALPHA * (target - row[action])

    def act_dqn(self, state):
        if random.random() < self.eps:
            return random.randrange(3)
        return int(np.argmax(self.online.q_values(state)))

    def train_dqn(self):
        self.env_steps += 1
        if (self.env_steps < WARMUP or self.buffer.n < BATCH_SIZE
                or self.env_steps % TRAIN_EVERY):
            return
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        best_next = np.argmax(self.online.forward(s2)[0], axis=1)
        q_next = self.target.forward(s2)[0][np.arange(BATCH_SIZE), best_next]
        targets = (r + GAMMA * (1.0 - d) * q_next).astype(np.float32)
        self.online.train_step(s, a, targets, LR)
        if self.env_steps % TARGET_SYNC == 0:
            self.target.copy_from(self.online)

    # ----------------------------------------------------------- rewards --
    def reward_blob(self, prev_h_norm):
        """Camera-only progress: growing blob height = getting closer.
        Being lost costs extra, which makes 'search until found' valuable -
        and re-acquiring the beacon after a search is rewarded automatically."""
        r = -STEP_PENALTY
        if self.blob is None:
            r -= SEARCH_PENALTY
        else:
            r += VIS_SHAPING * (self.blob[1] / self.cam_h - prev_h_norm)
        if max(self.ds[7].getValue(), self.ds[0].getValue()) > BUMP_THRESHOLD:
            r -= BUMP_PENALTY
        done = self.goal_distance() < GOAL_RADIUS
        if done:
            r += GOAL_REWARD
        return r, done

    def reward_pixels(self, prev_dist):
        r = 5.0 * (prev_dist - self.goal_distance()) - STEP_PENALTY
        if max(self.ds[7].getValue(), self.ds[0].getValue()) > BUMP_THRESHOLD:
            r -= BUMP_PENALTY
        done = self.goal_distance() < GOAL_RADIUS
        if done:
            r += GOAL_REWARD
        return r, done, self.goal_distance()

    # ---------------------------------------------------------- episodes --
    def _new_episode(self, random_start=True):
        x, y, z = self.start_pos
        self.trans_field.setSFVec3f([x, y, z])
        angle = random.uniform(0, 2 * math.pi) if random_start else 0.0
        self.rot_field.setSFRotation([0, 1, 0, angle])
        self.me.resetPhysics()
        for m in self.motors:
            m.setVelocity(0.0)
        self.prev_dist = self.goal_distance()
        self.prev_h = 0.0
        self.steps = 0
        self.episode_reward = 0.0

    def _log(self, reached):
        header = not os.path.exists(self.log_file)
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(["episode", "steps", "reward",
                            "epsilon", "reached"])
            w.writerow([self.episode, self.steps,
                        round(self.episode_reward, 2),
                        round(self.eps, 3), int(reached)])

    def _end_episode(self, reached):
        self.episode += 1
        self._log(reached)
        if self.mode == "blob":
            self.eps = 0.0 if (self.args.episodes
                               and self.episode >= self.args.episodes) \
                else max(TB_EPS_MIN, self.eps * TB_EPS_DECAY)
        else:
            if not (self.args.episodes
                    and self.episode >= self.args.episodes):
                self.eps = max(EPS_END, self.eps * EPS_DECAY)
        print("%s [%s]: episode %d: %s after %d steps, reward=%.1f, eps=%.2f"
              % (self.getName(), self.mode, self.episode,
                 "GOAL" if reached else "timeout", self.steps,
                 self.episode_reward, self.eps), flush=True)
        if self.episode % SAVE_EVERY == 0:
            if self.mode == "blob":
                self._save_qtable()
            else:
                self._save_weights()
        self._new_episode()

    # ------------------------------------------------------------- loop ---
    def run(self):
        while self.step(self.ts) != -1:
            if self.mode == "blob":
                self._scan()
                obs = self.state_blob()
                action = self.choose_tabular(obs)
            else:
                obs = self.state_pixels()
                action = self.act_dqn(obs)
            self.drive(action)
            if self.step(self.ts) == -1:
                break
            if self.mode == "blob":
                self._scan()
                nobs = self.state_blob()
                r, done = self.reward_blob(self.prev_h)
                self.prev_h = (self.blob[1] / self.cam_h
                               if self.blob else 0.0)
                self.learn_tabular(obs, action, r, nobs, done)
            else:
                nobs = self.state_pixels()
                r, done, dist = self.reward_pixels(self.prev_dist)
                self.prev_dist = dist
                self.buffer.add(obs, action, r, nobs, float(done))
                self.train_dqn()
            self.episode_reward += r
            self.steps += 1
            if done or self.steps >= MAX_STEPS:
                self._end_episode(done)


if __name__ == "__main__":
    VisionEpuck().run()
