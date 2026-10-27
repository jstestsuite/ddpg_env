"""Plot training curves from a run directory.

    ros2 run ddpg_env plot <run_dir> [<run_dir> ...]

Writes <run_dir>/progress.png.
"""

import csv
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rolling(x, n=20):
    x = np.asarray(x, dtype=float)
    warnings.filterwarnings('ignore', 'Mean of empty slice')  # NaN losses before training starts
    return np.array([np.nanmean(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])


def plot_run(run_dir):
    with open(os.path.join(run_dir, 'episodes.csv')) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f'{run_dir}: no episodes yet')
        return
    ep = [int(r['episode']) for r in rows]
    col = {k: [float(r[k]) for r in rows]
           for k in ('score', 'goals', 'critic_loss', 'q_mean', 'noise')}
    collided = [r['outcome'] == 'collision' for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    panels = [
        (axes[0, 0], 'score', 'episode score'),
        (axes[0, 1], 'goals', 'goals per episode'),
        (axes[1, 0], 'critic_loss', 'critic loss'),
        (axes[1, 1], 'q_mean', 'mean Q'),
    ]
    for ax, key, title in panels:
        ax.plot(ep, col[key], alpha=0.25, color='C0')
        ax.plot(ep, rolling(col[key]), color='C0', label='20-episode mean')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0, 1].plot(ep, rolling(collided), color='C3', label='collision rate (20 ep)')
    axes[0, 1].legend(loc='upper left')
    axes[1, 0].set_yscale('symlog')
    for ax in axes[1]:
        ax.set_xlabel('episode')
    fig.suptitle(os.path.basename(os.path.normpath(run_dir)))
    fig.tight_layout()
    out = os.path.join(run_dir, 'progress.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for run_dir in sys.argv[1:]:
        plot_run(os.path.expanduser(run_dir))


if __name__ == '__main__':
    main()
