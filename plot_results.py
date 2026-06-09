"""
Plot training curves from log files.
Usage: python plot_results.py out-dir1 out-dir2 ...
"""
import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 12


def load_log(log_dir):
    log_files = sorted(glob.glob(os.path.join(log_dir, 'log_*.txt')))
    if not log_files:
        raise FileNotFoundError(f'no log file in {log_dir}')
    path = log_files[-1]

    data = {'step': [], 'val_loss': [], 'iter': [], 'loss': []}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7 or line.startswith('step,'):
                continue
            step, train_l, val_l, it, loss, t_ms, mfu = parts
            if step:
                data['step'].append(int(step))
                data['val_loss'].append(float(val_l))
            if it:
                data['iter'].append(int(it))
                data['loss'].append(float(loss))

    label = os.path.basename(log_dir).replace('out-wikitext103-', '').replace('out-', '')
    return label, data


def main():
    dirs = sys.argv[1:] if len(sys.argv) > 1 else glob.glob('out-wikitext103-*')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    labels, final_vals = [], []

    for i, d in enumerate(dirs):
        label, data = load_log(d)
        c = colors[i % len(colors)]
        labels.append(label)
        if data['val_loss']:
            final_vals.append(data['val_loss'][-1])
        else:
            final_vals.append(0)

        # right: training loss
        axes[1].plot(data['iter'], data['loss'], color=c, alpha=0.7, lw=0.8, label=label)

    # left: bar chart of final val loss
    x = np.arange(len(labels))
    bars = axes[0].bar(x, final_vals, color=colors[:len(labels)], edgecolor='white', width=0.6)

    # annotate bars
    for bar, val in zip(bars, final_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel('val loss')
    axes[0].set_title('Final Validation Loss')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(min(final_vals) - 0.1, max(final_vals) + 0.1)

    axes[1].set_xlabel('iter')
    axes[1].set_ylabel('train loss')
    axes[1].set_title('Training Loss')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'results_all.png'
    plt.savefig(out_path, dpi=150)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    main()
