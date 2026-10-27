"""Train (or test) the DDPG agent.

    ros2 run ddpg_env train --ros-args --params-file <config.yaml> [-p key:=value ...]
"""

import csv
import datetime
import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
import torch

from ddpg_env.agent import Agent, ReplayBuffer
from ddpg_env.env import TurtleBotEnv

DEFAULTS = {
    'algo': 'ddpg',              # 'ddpg' or 'td3'
    'mode': 'steer',             # 'steer' (fixed speed) or 'steer_speed'
    'train': True,               # False = run the loaded policy with no noise or learning
    'checkpoint': '',            # .pt file to load ('' = start fresh)
    'run_dir': '',               # '' = ~/ros2_ws/ddpg_runs/<mode>_<timestamp>
    'episodes': 1000,            # upper limit; plateau detection usually stops sooner
    'max_steps': 500,            # 100 s of sim time at 5 Hz
    'seed': -1,
    'device': 'auto',
    # sim (must match stage.launch.py)
    'stage': 1,
    'world_name': 'dqn',
    'robot_name': 'burger',
    # robot
    'fixed_linear': 0.15,        # m/s, steer mode only
    'max_linear': 0.22,          # burger hardware limit
    'max_angular': 1.5,
    'n_scan_bins': 24,
    'collision_distance': 0.15,
    'goal_distance': 0.2,
    # reward
    'progress_scale': 40.0,      # reward per metre of progress toward the goal
    'time_penalty': 0.05,        # per step, so dawdling costs something
    'goal_reward': 100.0,
    'collision_reward': -100.0,
    # DDPG
    'hidden': [256, 256],
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_delay': 2,           # TD3 only: critic updates per actor/target update
    'target_noise': 0.2,         # TD3 only: smoothing noise on target actions
    'target_noise_clip': 0.5,
    'batch_size': 128,
    'buffer_size': 200000,
    'warmup_steps': 1000,        # uniform-random actions before the policy takes over
    'updates_per_step': 1,
    'noise_start': 0.4,          # gaussian sigma in action units ([-1, 1])
    'noise_end': 0.05,
    'noise_decay_steps': 50000,
    'save_every': 10,
    # Stop early once training plateaus: the rolling average of goals per episode
    # (over plateau_window episodes) hasn't risen by at least
    # max(plateau_min_delta, plateau_min_rel * itself) in plateau_patience episodes.
    # Only checked after noise has fully decayed and plateau_min_episodes have run.
    'plateau_window': 50,
    'plateau_patience': 150,     # 0 disables early stopping
    'plateau_min_episodes': 150,
    'plateau_min_delta': 0.1,    # goals per episode
    'plateau_min_rel': 0.05,
}

CSV_FIELDS = ['episode', 'steps', 'total_steps', 'score', 'goals', 'outcome', 'noise',
              'critic_loss', 'actor_loss', 'q_mean', 'late_steps', 'wall_s']


def noise_sigma(cfg, total_steps):
    frac = min(1.0, max(0, total_steps - cfg['warmup_steps']) / cfg['noise_decay_steps'])
    return cfg['noise_start'] + frac * (cfg['noise_end'] - cfg['noise_start'])


def make_run_dir(cfg):
    run_dir = cfg['run_dir'] or os.path.expanduser(os.path.join(
        '~/ros2_ws/ddpg_runs',
        f"{cfg['algo']}_{cfg['mode']}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        + ('' if cfg['train'] else '_test')))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    return run_dir


def main(args=None):
    rclpy.init(args=args)
    # Parameters live on the env node so a single --params-file configures everything
    env = TurtleBotEnv(DEFAULTS)
    cfg = env.cfg
    cfg['hidden'] = [int(h) for h in cfg['hidden']]

    executor = SingleThreadedExecutor()
    executor.add_node(env)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        except Exception:
            # Ctrl+C can invalidate the context mid-wait, which raises a different error
            if rclpy.ok():
                raise

    threading.Thread(target=spin, daemon=True).start()
    log = env.get_logger()

    if cfg['seed'] >= 0:
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
    device = cfg['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"algo={cfg['algo']} mode={cfg['mode']} train={cfg['train']} device={device}"
             + (f' ({torch.cuda.get_device_name(0)})' if device.startswith('cuda') else ''))

    agent = Agent(env.state_dim, env.action_dim, cfg['algo'], cfg['hidden'], cfg['actor_lr'],
                  cfg['critic_lr'], cfg['gamma'], cfg['tau'], cfg['policy_delay'],
                  cfg['target_noise'], cfg['target_noise_clip'], device)
    total_steps = 0
    start_episode = 0
    if cfg['checkpoint']:
        path = os.path.expanduser(cfg['checkpoint'])
        ckpt_cfg = torch.load(path, map_location='cpu')['extra'].get('cfg', {})
        for key in ('mode', 'n_scan_bins', 'hidden'):
            if key in ckpt_cfg and ckpt_cfg[key] != cfg[key]:
                raise SystemExit(f'{path} was trained with {key}={ckpt_cfg[key]}, but this run '
                                 f'has {key}={cfg[key]}; use the matching config file')
        extra = agent.load(path, actor_only=not cfg['train'])
        total_steps = extra.get('total_steps', 0) if cfg['train'] else 0
        start_episode = extra.get('episode', -1) + 1 if cfg['train'] else 0
        log.info(f"loaded {cfg['checkpoint']} (episode {start_episode}, step {total_steps})")

    run_dir = make_run_dir(cfg)
    log.info(f'logging to {run_dir}')
    buffer = ReplayBuffer(env.state_dim, env.action_dim, cfg['buffer_size'])
    csv_file = open(os.path.join(run_dir, 'episodes.csv'), 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    losses = []

    def train_step():
        if not cfg['train'] or buffer.size < max(cfg['batch_size'], cfg['warmup_steps']):
            return
        for _ in range(cfg['updates_per_step']):
            losses.append(agent.update(buffer, cfg['batch_size']))

    def save(name, episode):
        agent.save(os.path.join(run_dir, name),
                   {'episode': episode, 'total_steps': total_steps, 'cfg': cfg})

    recent_goals = []
    best_avg_goals = -1.0
    ref_avg_goals = float('-inf')
    best_episode = start_episode
    t0 = time.monotonic()
    episode = start_episode
    try:
        for episode in range(start_episode, start_episode + cfg['episodes']):
            state = env.reset()
            score, goals, outcome = 0.0, 0, 'timeout'
            losses.clear()
            sigma = noise_sigma(cfg, total_steps) if cfg['train'] else 0.0
            for t in range(cfg['max_steps']):
                if cfg['train'] and total_steps < cfg['warmup_steps']:
                    action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
                else:
                    action = agent.act(state, sigma)
                next_state, reward, terminal, event = env.step(action, during_wait=train_step)
                if cfg['train']:
                    buffer.add(state, action, reward, next_state, terminal)
                score += reward
                total_steps += 1
                if event == 'collision':
                    outcome = 'collision'
                    break
                if event == 'goal':
                    goals += 1
                    state = env.next_goal()
                else:
                    state = next_state
            steps = t + 1

            l = np.array(losses) if losses else np.full((1, 3), np.nan)
            writer.writerow({
                'episode': episode, 'steps': steps, 'total_steps': total_steps,
                'score': round(score, 2), 'goals': goals, 'outcome': outcome,
                'noise': round(sigma, 4), 'critic_loss': round(float(l[:, 0].mean()), 4),
                'actor_loss': round(float(l[:, 1].mean()), 4),
                'q_mean': round(float(l[:, 2].mean()), 3), 'late_steps': env.late_steps,
                'wall_s': round(time.monotonic() - t0, 1),
            })
            csv_file.flush()

            window = cfg['plateau_window']
            recent_goals = (recent_goals + [goals])[-window:]
            avg_goals = float(np.mean(recent_goals))
            log.info(f'ep {episode:4d} | {outcome:9s} | steps {steps:3d} | goals {goals:2d} '
                     f'| score {score:8.1f} | avg goals ({window}) {avg_goals:5.2f} '
                     f'| noise {sigma:.3f} | total steps {total_steps} '
                     f'| {total_steps / max(time.monotonic() - t0, 1e-6):.0f} steps/s'
                     + (f' | late steps {env.late_steps}' if env.late_steps else ''))

            if not cfg['train']:
                continue
            if (episode + 1) % cfg['save_every'] == 0:
                save('latest.pt', episode)
            if len(recent_goals) < window:
                continue
            if avg_goals > best_avg_goals:
                best_avg_goals = avg_goals
                save('best.pt', episode)
            # "Meaningful" is measured from the last meaningful improvement, so a slow
            # steady climb still counts even if each new best is only slightly higher
            if avg_goals - ref_avg_goals >= max(cfg['plateau_min_delta'],
                                                cfg['plateau_min_rel'] * ref_avg_goals):
                ref_avg_goals = avg_goals
                best_episode = episode
            # Plateau: noise has finished decaying (so the measured performance is the
            # near-final policy) and the rolling average hasn't improved meaningfully
            # in plateau_patience episodes.
            noise_done = total_steps >= cfg['warmup_steps'] + cfg['noise_decay_steps']
            stalled = episode - best_episode
            if (cfg['plateau_patience'] > 0 and noise_done
                    and episode + 1 >= cfg['plateau_min_episodes']
                    and stalled >= cfg['plateau_patience']):
                log.info(f'stopping: avg goals has not improved meaningfully for {stalled} '
                         f'episodes (last at episode {best_episode}: {ref_avg_goals:.2f}; '
                         f'best {best_avg_goals:.2f})')
                break
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        log.error(f'stopping run: {e}')
    finally:
        if rclpy.ok():
            env.stop()
        if cfg['train']:
            save('latest.pt', episode)
            log.info(f'saved {os.path.join(run_dir, "latest.pt")}')
        csv_file.close()
        executor.shutdown()
        env.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
