"""q_visualizer.py - live Q-table monitor rendered on a Webots Display.

Learners need --telemetry in their controllerArgs. The monitor's Receiver
listens on channel 1 (the e-pucks' default emitter channel).
Panels: 8 cols = goal-bearing sector (4 = straight ahead),
        8 rows = obstacle pattern (front/left/right bits).
Arrow = greedy action (blue FORWARD, orange LEFT, green RIGHT);
brightness = |Q|; grey = never visited. Watching grey turn into arrows
IS watching the robots learn.
"""
import json

from controller import Robot

PANELS = 3
CELL = 18
ACTION_COLORS = [0x3B82F6, 0xF59E0B, 0x22C55E]   # FORWARD LEFT RIGHT
BG, GRID, TEXT, DIM = 0x10141A, 0x232B36, 0xDDE3EC, 0x66707F


class QVisualizer(Robot):
    def __init__(self):
        super().__init__()
        self.ts = int(self.getBasicTimeStep())
        self.display = self.getDevice("display")
        self.w, self.h = self.display.getWidth(), self.display.getHeight()
        try:
            self.display.setFont("Lucida Console", 11, True)
        except Exception:
            pass
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.ts)
        try:
            self.receiver.setChannel(-1)          # hear every channel if allowed
        except Exception:
            pass                                  # fall back to world channel
        self.robots = {}                          # name -> {"stats","q"}
        self.redraw_in = 0

    @staticmethod
    def _scale(color, factor):
        r = min(255, int(((color >> 16) & 0xFF) * factor))
        g = min(255, int(((color >> 8) & 0xFF) * factor))
        b = min(255, int((color & 0xFF) * factor))
        return (r << 16) | (g << 8) | b

    def poll(self):
        while self.receiver.getQueueLength() > 0:
            try:
                msg = json.loads(self.receiver.getData().decode("utf-8"))
                if msg.get("t") == "telemetry":
                    self.robots[msg.get("name", "?")] = {
                        "stats": msg,
                        "q": {int(k): v for k, v in msg.get("q", {}).items()}}
            except Exception:
                pass
            self.receiver.nextPacket()

    def draw_arrow(self, cx, cy, action, color, f):
        d = self.display
        d.setColor(self._scale(color, f))
        if action == 0:                                   # FORWARD = up
            d.drawLine(cx, cy - 6, cx, cy + 6)
            d.drawLine(cx - 3, cy - 2, cx, cy - 6)
            d.drawLine(cx + 3, cy - 2, cx, cy - 6)
        elif action == 1:                                 # LEFT
            d.drawLine(cx - 6, cy, cx + 6, cy)
            d.drawLine(cx - 2, cy - 3, cx - 6, cy)
            d.drawLine(cx - 2, cy + 3, cx - 6, cy)
        else:                                             # RIGHT
            d.drawLine(cx - 6, cy, cx + 6, cy)
            d.drawLine(cx + 2, cy - 3, cx + 6, cy)
            d.drawLine(cx + 2, cy + 3, cx + 6, cy)

    def draw_panel(self, i, name):
        d = self.display
        st = self.robots[name]["stats"]
        q = self.robots[name]["q"]
        x0, y0 = 8 + i * 158, 34
        d.setColor(TEXT)
        d.drawText(("%s [%s]" % (name[:13], st.get("algo", "?")))[:18],
                   x0, y0)
        d.setColor(DIM)
        d.drawText("col=goal bearing (4=ahead)", x0, y0 + 12)
        oy = y0 + 26
        d.setColor(GRID)                       # grid + unexplored cells first
        for r in range(8):
            for c in range(8):
                d.drawRectangle(x0 + c * CELL, oy + r * CELL, CELL - 1,
                                CELL - 1)
        vals = [abs(v) for row in q.values() for v in row] or [1.0]
        vmax = max(vals) or 1.0
        for state, row in q.items():            # then greedy-action arrows
            best = max(range(len(row)), key=lambda a: row[a])
            f = 0.35 + 0.65 * (abs(row[best]) / vmax)
            r, c = state & 7, state >> 3        # row=obstacles, col=bearing
            self.draw_arrow(x0 + c * CELL + CELL // 2,
                            oy + r * CELL + CELL // 2,
                            best, ACTION_COLORS[best], f)
        yy = oy + 8 * CELL + 6
        d.setColor(TEXT)
        d.drawText("ep %-4d steps %-4d" % (st.get("episode", 0),
                                           st.get("steps", 0)), x0, yy)
        d.drawText("eps %.2f  rw %6.1f"
                   % (st.get("epsilon", 0.0), st.get("reward", 0.0)),
                   x0, yy + 14)
        tag = "GOAL" if st.get("reached") else " .."
        d.drawText("states %2d/64  %s" % (st.get("states", len(q)), tag),
                   x0, yy + 28)

    def draw(self):
        d = self.display
        d.setColor(BG)
        d.fillRectangle(0, 0, self.w, self.h)
        d.setColor(TEXT)
        d.drawText("LIVE Q-MONITOR", 8, 8)
        d.setColor(DIM)
        d.drawText("brightness=|Q|  gray=unvisited", 130, 8)
        for i, name in enumerate(list(self.robots)[:PANELS]):
            self.draw_panel(i, name)
        if len(self.robots) > PANELS:
            d.setColor(DIM)
            d.drawText("+ %d more robot(s)" % (len(self.robots) - PANELS),
                       8, self.h - 14)

    def run(self):
        while self.step(self.ts) != -1:
            self.poll()
            self.redraw_in -= self.ts
            if self.redraw_in <= 0:
                self.draw()
                self.redraw_in = 500              # ms of simulated time


if __name__ == "__main__":
    QVisualizer().run()
