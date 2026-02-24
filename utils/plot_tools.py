import matplotlib.pyplot as plt
import os


def plot_cooperators_ratio(cooperators_frac, dirName, sim, b):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cooperators_frac)), cooperators_frac, marker='o', markersize=2, linestyle='-', color='b')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Fraction of cooperators', fontsize=14)
    plt.title(f'Evolution of cooperation over time (b = {b:.2f})', fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(dirName, f'cooperators_frac_b{b:.2f}_sim{sim:04d}.png'), dpi=300)
    plt.show()
