"""qlearning_agent.py v1.2 (FINAL) - self-learning e-puck for Webots.

  --algo=qlearning|sarsa   off-policy vs on-policy TD updates
  --share                  count-weighted consensus Q-table sharing (radio)
  --telemetry              broadcast stats+Q-table each episode (q_visualizer)
  --seed / --episodes      reproducibility / greedy-after-N

Q-table file format v2: {"q": {state: [q0,q1,q2]}, "n": {state: [n0,n1,n2]}}
Legacy plain {state: [...]} files still load (visit counts start at zero).

State: integer 0..63 encoding (front,left,right obstacles + 8 goal sectors).
Actions: FORWARD, LEFT, RIGHT. CSV log: progress_<name>.csv
"""

import argparse
import csv
import json
import math
import os
import random

from controller import Supervisor

# ------------------------------------------------------------- tune me ----
ALPHA = 0.2             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97    # per episode

MAX_SPEED = 6.28        # e-puck wheel speed limit (rad/s)
TURN_SLOWDOWN = 0.2

OBSTACLE_THRESHOLD = 600
BUMP_THRESHOLD = 1500
GOAL_RADIUS = 0.15
SHAPING = 5.0           # reward per metre of goal progress
STEP_PENALTY = 0.01
BUMP_PENALTY = 0.3
GOAL_REWARD = 10.0

MAX_STEPS = 800
SAVE_EVERY = 10
COUNT_CAP = 50          # max merge weight of any single visit count

ACTIONS = ["FORWARD", "LEFT", "RIGHT"]


class LearningEpuck(Supervisor):
    def __init__(self):
        super().__init__()
        p = argparse.ArgumentParser()
        p.add_argument("--goal", default="1.2,1.2", help="goal x,z (m)")
        p.add_argument("--qtable", default="qtable.json")
        p.add_argument("--algo", choices=["qlearning", "sarsa"],
                       default="qlearning")
        p.add_argument("--share", action="store_true",
                       help="team learning via radio")
        p.add_argument("--telemetry", action="store_true",
                       help="broadcast stats+Q-table to q_visualizer")
        p.add_argument("--episodes", type=int, default=0,
                       help="go fully greedy after N episodes (0 = never)")
        p.add_argument("--seed", type=int, default=None)
        self.args = p.parse_args()
        self.algo = self.args.algo
        if self.args.seed is not None:
            random.seed(self.args.seed)

        self.goal = tuple(float(x) for x in self.args.goal.split(","))
        folder = os.path.dirname(os.path.abspath(__file__))
        self.qtable_file = os.path.join(folder, self.args.qtable)
        self.log_file = os.path.join(folder,
                                     "progress_%s.csv" % self.getName())

        self.control_ts = 4 * int(self.getBasicTimeStep())
        self.ds = [self.getDevice("ps%d" % i) for i in range(8)]
        for s in self.ds:
            s.enable(self.control_ts)
        self.motors = [self.getDevice("left wheel motor"),
                       self.getDevice("right wheel motor")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # radio (e-puck range & bearing board)
        self.emitter = self.getDevice("emitter")
        self.receiver = self.getDevice("receiver")
        self.radio = self.emitter is not None and self.receiver is not None
        if (self.args.share or self.args.telemetry) and not self.radio:
            print("%s: emitter/receiver unavailable - --share/--telemetry "
                  "disabled" % self.getName(), flush=True)
            self.args.share = self.args.telemetry = False
        if self.radio:
            self.receiver.enable(self.control_ts)
            try:
                self.emitter.setRange(-1)          # infinite range
            except Exception:
                pass

        self.me = self.getSelf()
        self.trans_field = self.me.getField("translation")
        self.rot_field = self.me.getField("rotation")
        self.start_pos = self.trans_field.getSFVec3f()

        self.q, self.counts = self._load_qtable()
        self.epsilon = EPSILON_START
        self.episode = 0
        self._new_episode(random_start=False)

    # ----------------------------------------------- persistence & merge --
    def _load_qtable(self):
        q, n = {}, {}
        try:
            with open(self.qtable_file) as f:
                data = json.load(f)
            if isinstance(data, dict) and "q" in data:      # v1.2 format
                q = {int(k): [float(x) for x in v]
                     for k, v in data["q"].items()}
                n = {int(k): [float(x) for x in v]
                     for k, v in data.get("n", {}).items()}
            else:                                           # legacy format
                q = {int(k): [float(x) for x in v]
                     for k, v in data.items()}
        except Exception:
            pass
        return q, n

    def _save_qtable(self):
        with open(self.qtable_file, "w") as f:
            json.dump({"q": self.q, "n": self.counts}, f)

    def _qrow(self, state):
        return self.q.setdefault(state, [0.0] * len(ACTIONS))

    def _crow(self, state):
        return self.counts.setdefault(state, [0.0] * len(ACTIONS))

    def _merge_table(self, other_q, other_n):
        """Count-weighted (federated-style) average. Each robot's opinion is
        weighted by min(visit_count, COUNT_CAP); the cap stops one long-running
        robot from permanently drowning out fresh experience."""
        for key, orow in other_q.items():
            state = int(key)
            qrow, nrow = self._qrow(state), self._crow(state)
            on = other_n.get(str(state), [0.0] * len(ACTIONS))
            for a in range(len(ACTIONS)):
                w_mine = min(nrow[a], COUNT_CAP)
                w_other = min(float(on[a]), COUNT_CAP)
                if w_mine + w_other <= 0.0:
                    continue
                qrow[a] = (w_mine * qrow[a] + w_other * float(orow[a])) \
                    / (w_mine + w_other)
                nrow[a] += float(on[a])

    # ------------------------------------------------------------ radio ---
    def _broadcast_qtable(self):
        self.emitter.send(json.dumps(
            {"t": "qtable", "name": self.getName(),
             "q": {str(k): v for k, v in self.q.items()},
             "n": {str(k): v for k, v in self.counts.items()}}
        ).encode("utf-8"))

    def _send_telemetry(self, reached):
        self.emitter.send(json.dumps(
            {"t": "telemetry", "name": self.getName(), "algo": self.algo,
             "episode": self.episode, "steps": self.steps,
             "reward": round(self.episode_reward, 2),
             "epsilon": round(self.epsilon, 3), "reached": int(reached),
             "states": len(self.q),
             "q": {str(k): v for k, v in self.q.items()}}
        ).encode("utf-8"))

    def _receive_messages(self):
        merged = 0
        while self.receiver.getQueueLength() > 0:
            try:
                msg = json.loads(self.receiver.getData().decode("utf-8"))
                if (msg.get("t") == "qtable"
                        and msg.get("name") != self.getName()):
                    self._merge_table(msg.get("q", {}), msg.get("n", {}))
                    merged += 1
                # telemetry packets are ignored by learners
            except Exception:
                pass                                   # ignore torn packets
            self.receiver.nextPacket()
        return merged

    # ------------------------------------------------------------ senses --
    def pose(self):
        pos = self.me.getPosition()
        m = self.me.getOrientation()
        return pos, math.atan2(m[2], m[8])     # forward = local +z

    def state(self):
        """Integer state id: bits 0-2 = obstacles, bits 3-5 = goal sector."""
        v = [s.getValue() for s in self.ds]
        front = int(max(v[7], v[0]) > OBSTACLE_THRESHOLD)
        left = int(max(v[6], v[5]) > OBSTACLE_THRESHOLD)
        right = int(max(v[1], v[2]) > OBSTACLE_THRESHOLD)
        pos, heading = self.pose()
        bearing = math.atan2(self.goal[0] - pos[0], self.goal[2] - pos[2])
        rel = (bearing - heading + math.pi) % (2.0 * math.pi) - math.pi
        sector = int((rel + math.pi) / (math.pi / 4.0)) % 8
        return front | (left << 1) | (right << 2) | (sector << 3)

    def goal_distance(self):
        pos, _ = self.pose()
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

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        row = self._qrow(state)
        best = max(row)
        return random.choice([a for a, v in enumerate(row) if v == best])

    def reward(self, previous_distance):
        r = SHAPING * (previous_distance - self.goal_distance()) - STEP_PENALTY
        if max(self.ds[7].getValue(), self.ds[0].getValue()) > BUMP_THRESHOLD:
            r -= BUMP_PENALTY
        done = self.goal_distance() < GOAL_RADIUS
        if done:
            r += GOAL_REWARD
        return r, done

    def learn(self, state, action, r, next_state, done, next_action=None):
        self._crow(state)[action] += 1.0          # visit counting for merges
        if done:
            target = r
        elif self.algo == "sarsa" and next_action is not None:
            target = r + GAMMA * self._qrow(next_state)[next_action]
        else:                                     # Q-learning: max over next
            target = r + GAMMA * max(self._qrow(next_state))
        row = self._qrow(state)
        row[action] += ALPHA * (target - row[action])

    # --------------------------------------------------------- episodes ---
    def _new_episode(self, random_start=True):
        x, y, z = self.start_pos
        self.trans_field.setSFVec3f([x, y, z])
        angle = random.uniform(0, 2 * math.pi) if random_start else 0.0
        self.rot_field.setSFRotation([0, 1, 0, angle])
        self.me.resetPhysics()
        for m in self.motors:
            m.setVelocity(0.0)
        self.prev_distance = self.goal_distance()
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
                        round(self.epsilon, 3), int(reached)])

    def _end_episode(self, reached):
        self.episode += 1
        merged = 0
        if self.radio:
            if self.args.share:
                merged = self._receive_messages()
                self._broadcast_qtable()
            else:
                self._receive_messages()          # just drain the queue
        if self.args.telemetry:
            self._send_telemetry(reached)
        self._log(reached)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if self.args.episodes and self.episode >= self.args.episodes:
            self.epsilon = 0.0
        print("%s [%s]: episode %d: %s after %d steps, reward=%.1f, "
              "eps=%.2f, states=%d/64, heard=%d"
              % (self.getName(), self.algo, self.episode,
                 "GOAL" if reached else "timeout", self.steps,
                 self.episode_reward, self.epsilon, len(self.q), merged),
              flush=True)
        if self.episode % SAVE_EVERY == 0:
            self._save_qtable()
        self._new_episode()

    # ------------------------------------------------------------- loop ---
    def run(self):
        while self.step(self.control_ts) != -1:
            state = self.state()
            action = self.choose_action(state)
            self.drive(action)
            if self.step(self.control_ts) == -1:
                break
            next_state = self.state()
            next_action = self.choose_action(next_state)   # needed by SARSA
            r, done = self.reward(self.prev_distance)
            self.learn(state, action, r, next_state, done, next_action)
            self.episode_reward += r
            self.prev_distance = self.goal_distance()
            self.steps += 1
            state, action = next_state, next_action
            if done or self.steps >= MAX_STEPS:
                self._end_episode(done)


if __name__ == "__main__":
    LearningEpuck().run()
