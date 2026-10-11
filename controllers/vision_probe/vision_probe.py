"""vision_probe.py - orbiting camera inspection station (supervisor).

Circles the beacon and renders what the camera sees on its Display,
outlining pixels that match the goal colour. Tuning tool, not a learner.
SPACE = pause/resume the orbit.
"""
import argparse
import math

import numpy as np
from controller import Supervisor

SCALE = 5


class Probe(Supervisor):
    def __init__(self):
        super().__init__()
        p = argparse.ArgumentParser()
        p.add_argument("--rgb", default="217,178,26", help="goal colour")
        p.add_argument("--tol", type=int, default=60, help="per-channel +/-")
        p.add_argument("--center", default="0,0", help="beacon x,z")
        p.add_argument("--radius", type=float, default=1.4)
        p.add_argument("--height", type=float, default=0.25)
        p.add_argument("--speed", type=float, default=0.25)   # rad/s
        self.args = p.parse_args()
        self.rgb = tuple(int(v) for v in self.args.rgb.split(","))
        self.cx, self.cz = (float(v) for v in self.args.center.split(","))

        self.ts = int(self.getBasicTimeStep())
        self.cam = self.getDevice("camera")
        self.cam.enable(self.ts)
        self.w, self.h = self.cam.getWidth(), self.cam.getHeight()
        self.display = self.getDevice("display")
        try:
            self.display.setFont("Lucida Console", 9, True)
        except Exception:
            pass
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.ts)

        self.me = self.getSelf()
        self.trans = self.me.getField("translation")
        self.rot = self.me.getField("rotation")
        self.angle, self.paused, self.frame = 0.0, False, 0
        self.img = None

    def _scan(self):
        self.img = self.cam.getImage()
        buf = np.frombuffer(self.img, dtype=np.uint8).astype(np.int16)
        buf = buf.reshape(self.h, self.w, 4)
        r, g, b = buf[:, :, 2], buf[:, :, 1], buf[:, :, 0]
        tr, tg, tb = self.rgb
        mask = ((np.abs(r - tr) < self.args.tol)
                & (np.abs(g - tg) < self.args.tol)
                & (np.abs(b - tb) < self.args.tol))
        ys, xs = np.nonzero(mask)
        return ys, xs

    def _place(self):
        x = self.cx + self.args.radius * math.sin(self.angle)
        z = self.cz + self.args.radius * math.cos(self.angle)
        self.trans.setSFVec3f([x, self.args.height, z])
        self.rot.setSFRotation([0, 1, 0, self.angle + math.pi])  # face beacon

    def _draw(self, ys, xs):
        d = self.display
        buf = np.frombuffer(self.img,
                            dtype=np.uint8).reshape(self.h, self.w, 4)
        for y in range(self.h):
            row = buf[y]
            for x in range(self.w):
                px = row[x]
                d.setColor((int(px[2]) << 16) | (int(px[1]) << 8) | int(px[0]))
                d.fillRectangle(x * SCALE, y * SCALE, SCALE - 1, SCALE - 1)
        if ys.size:                                     # outline matches
            d.setColor(0xFF00FF)
            step = max(1, ys.size // 300)
            for y, x in zip(ys[::step], xs[::step]):
                d.fillRectangle(int(x) * SCALE - 1, int(y) * SCALE - 1,
                                SCALE + 1, SCALE + 1)
        d.setColor(0x000000)
        d.fillRectangle(0, self.h * SCALE, self.w * SCALE, 34)
        d.setColor(0xFFFFFF)
        if ys.size:
            cx = float(xs.mean()) / self.w
            sector = min(7, int(cx * 8))
            d.drawText("blob cx=%.2f sector=%d h=%dpx n=%d"
                       % (cx, sector, ys.max() - ys.min() + 1, ys.size),
                       4, self.h * SCALE + 4)
        else:
            d.drawText("NO BLOB (n=0) - fix --rgb / --tol",
                       4, self.h * SCALE + 4)
        d.drawText("target %s  tol %d   [SPACE]=%s"
                   % (self.rgb, self.args.tol,
                      "resume" if self.paused else "pause"),
                   4, self.h * SCALE + 18)

    def run(self):
        while self.step(self.ts) != -1:
            key = self.keyboard.getKey()
            if key == ord(" "):
                self.paused = not self.paused
            if not self.paused:
                self.angle += self.args.speed * self.ts / 1000.0
                self._place()
            ys, xs = self._scan()
            self._draw(ys, xs)
            self.frame += 1
            if self.frame % 30 == 0:
                print("probe: %s" % ("n=%d" % ys.size if ys.size
                                     else "no blob"), flush=True)


if __name__ == "__main__":
    Probe().run()
