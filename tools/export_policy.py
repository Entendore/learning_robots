#!/usr/bin/env python3
"""export_policy.py - bake a learned Q-table into a standalone greedy controller.

Usage:
  python3 tools/export_policy.py controllers/qlearning_agent/qtable_robot1.json \\
      --name greedy_robot1 --goal 1.2,1.2

Creates controllers/greedy_robot1/controller.py. Set the exported robot's
controller to "greedy_robot1" in any world. For a real e-puck, use Webots'
cross-compilation tools on the generated file.
"""
import argparse
import json
import os

TEMPLATE = '''"""__NAME__ - greedy policy exported from __SOURCE__.

State encoding and actions match qlearning_agent v1.2. Odometry starts at
(0, 0, heading 0) at power-on, so GOAL is RELATIVE TO THE START POSE with
world-aligned axes. Sensor thresholds tuned in simulation may need
retuning on real hardware (measure with a small test script first).
"""
import math

from controller import Robot

GOAL = (__GX__, __GZ__)
Q = __Q__
DEFAULT_ACTION = 0                       # FORWARD for never-visited states

MAX_SPEED = 6.28
TURN_SLOWDOWN = 0.2
OBSTACLE_THRESHOLD = 600
WHEEL_RADIUS = 0.021                     # metres - match your robot!
AXLE_LENGTH = 0.053                      # metres - match your robot!


class GreedyEpuck(Robot):
    def __init__(self):
        super().__init__()
        self.ts = 4 * int(self.getBasicTimeStep())
        self.ds = [self.getDevice("ps%d" % i) for i in range(8)]
        for s in self.ds:
            s.enable(self.ts)
        self.motors = [self.getDevice("left wheel motor"),
                       self.getDevice("right wheel motor")]
        self.encoders = [self.getDevice("left wheel sensor"),
                         self.getDevice("right wheel sensor")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        for e in self.encoders:
            e.enable(self.ts)
        self.prev_enc = [e.getValue() for e in self.encoders]
        self.x = self.z = self.theta = 0.0

    def odometry(self):
        dl = self.encoders[0].getValue() - self.prev_enc[0]
        dr = self.encoders[1].getValue() - self.prev_enc[1]
        self.prev_enc = [self.encoders[0].getValue(),
                         self.encoders[1].getValue()]
        dt = self.ts / 1000.0
        vl = WHEEL_RADIUS * dl / dt
        vr = WHEEL_RADIUS * dr / dt
        v = 0.5 * (vl + vr)
        w = (vl - vr) / AXLE_LENGTH      # +w = nose toward local +x
        self.theta += w * dt
        self.x += v * math.sin(self.theta) * dt
        self.z += v * math.cos(self.theta) * dt

    def state(self):
        front = int(max(self.ds[7].getValue(),
                        self.ds[0].getValue()) > OBSTACLE_THRESHOLD)
        left = int(max(self.ds[6].getValue(),
                       self.ds[5].getValue()) > OBSTACLE_THRESHOLD)
        right = int(max(self.ds[1].getValue(),
                        self.ds[2].getValue()) > OBSTACLE_THRESHOLD)
        bearing = math.atan2(GOAL[0] - self.x, GOAL[1] - self.z)
        rel = (bearing - self.theta + math.pi) % (2 * math.pi) - math.pi
        sector = int((rel + math.pi) / (math.pi / 4)) % 8
        return front | (left << 1) | (right << 2) | (sector << 3)

    def run(self):
        while self.step(self.ts) != -1:
            self.odometry()
            s = self.state()
            action = max(range(3), key=lambda a: Q[s][a]) if s in Q \\
                else DEFAULT_ACTION
            l = r = MAX_SPEED
            if action == 1:
                l = MAX_SPEED * TURN_SLOWDOWN
            elif action == 2:
                r = MAX_SPEED * TURN_SLOWDOWN
            self.motors[0].setVelocity(l)
            self.motors[1].setVelocity(r)


if __name__ == "__main__":
    GreedyEpuck().run()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qtable", help="path to qtable_*.json (any version)")
    ap.add_argument("--name", required=True, help="controller folder name")
    ap.add_argument("--goal", default="1.2,1.2",
                    help="goal x,z relative to the robot's start pose")
    args = ap.parse_args()

    with open(args.qtable) as f:
        data = json.load(f)
    if isinstance(data, dict) and "q" in data:      # v1.2 format
        data = data["q"]
    q = {int(k): [round(float(x), 3) for x in v] for k, v in data.items()}
    gx, gz = (float(t) for t in args.goal.split(","))

    out_dir = os.path.join("controllers", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "controller.py")
    with open(out, "w") as f:
        f.write(TEMPLATE
                .replace("__NAME__", args.name)
                .replace("__SOURCE__", os.path.basename(args.qtable))
                .replace("__GX__", repr(gx))
                .replace("__GZ__", repr(gz))
                .replace("__Q__", repr(q)))

    print("wrote %s  (%d/64 states covered)" % (out, len(q)))
    print("next: set controller \"%s\" on an E-puck in your world."
          % args.name)


if __name__ == "__main__":
    main()
