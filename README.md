# Learning Robots — Webots Reinforcement Learning Suite

E-puck robots that learn **from scratch** — by trial, error and rewards — to
reach goals and avoid obstacles. One consistent state/action/reward language
across the whole suite: mix any robot, world and algorithm.

## Feature map

| Feature | Where | Concept |
|---|---|---|
| Tabular Q-learning | `qlearning_agent` | reward shaping, ε-greedy, state design |
| Q-learning vs SARSA | `--algo=...` + `benchmark_world` | off-policy vs on-policy |
| Team learning | `--share` | count-weighted consensus over the radio |
| Double-DQN (numpy only) | `dqn_agent` | function approximation, replay, target nets |
| Live Q-visualizer | `q_visualizer` + `--telemetry` | interpretability |
| Vision: blob homing | `vision_agent --mode=blob` | visual state, camera-only reward |
| Vision: pixels DQN | `vision_agent --mode=pixels` | end-to-end deep RL |
| Learning curves | `tools/plot_progress.py` | experiment analysis |
| Sim-to-real export | `tools/export_policy.py` | deployment, odometry |

## Requirements

- Webots **R2022b or newer** (R2023b recommended)
- Python 3 with **numpy** (DQN + vision agents); matplotlib for plots
- If your Webots names the robot `EPuck` (pre-R2022):
  `sed -i 's/E-puck/EPuck/g' worlds/*.wbt`

## Quick start

```bash
webots worlds/learning_robots.wbt        # train 3 team learners + live Q monitor
webots worlds/benchmark_world.wbt        # Q-learning vs SARSA (same seed)
webots worlds/maze_world.wbt             # maze with dead ends
webots worlds/vision_check.wbt           # calibrate beacon colours first
webots worlds/vision_world.wbt           # camera learners
webots worlds/learning_robots_dqn.wbt    # Deep Q-Network agents
