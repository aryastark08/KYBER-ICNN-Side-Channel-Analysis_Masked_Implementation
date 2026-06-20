import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load results ──────────────────────────────────────────────────────────────
results = np.load('all_bits_results.npy', allow_pickle=True).item()

# ── Extract data ──────────────────────────────────────────────────────────────
bits        = list(range(8))
train_acc   = [results[b]['train_acc'] * 100  for b in bits]
atk_n1      = [results[b]['attack_n1'] * 100  for b in bits]
atk_n8      = [results[b]['attack_n8'] * 100  for b in bits]
atk_n256    = [results[b]['attack_n256'] * 100 for b in bits]

# ── Plot setup ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

x      = np.arange(len(bits))
width  = 0.20
offset = [-1.5, -0.5, 0.5, 1.5]

colors = {
    'train' : '#3266AD',
    'n1'    : '#1D9E75',
    'n8'    : '#D85A30',
    'n256'  : '#888780',
}

bars_train = ax.bar(x + offset[0] * width, train_acc,  width, label='Training accuracy',     color=colors['train'], zorder=3)
bars_n1    = ax.bar(x + offset[1] * width, atk_n1,     width, label='Attack N=1 (single)',   color=colors['n1'],    zorder=3)
bars_n8    = ax.bar(x + offset[2] * width, atk_n8,     width, label='Attack N=8',            color=colors['n8'],    zorder=3)
bars_n256  = ax.bar(x + offset[3] * width, atk_n256,   width, label='Attack N=256',          color=colors['n256'],  zorder=3)

# ── Random baseline ───────────────────────────────────────────────────────────
ax.axhline(y=50, color='#E24B4A', linewidth=1.5,
           linestyle='--', zorder=4, label='Random baseline (50%)')

# ── Annotations for best results ──────────────────────────────────────────────
best_n1_val  = max(atk_n1)
best_n1_bit  = atk_n1.index(best_n1_val)
best_n8_val  = max(atk_n8)
best_n8_bit  = atk_n8.index(best_n8_val)

ax.annotate(f'{best_n1_val:.1f}%\nbest N=1',
            xy=(best_n1_bit + offset[1] * width, best_n1_val),
            xytext=(best_n1_bit + offset[1] * width + 0.3, best_n1_val + 4),
            fontsize=8, color=colors['n1'],
            arrowprops=dict(arrowstyle='->', color=colors['n1'], lw=1),
            ha='center')

ax.annotate(f'{best_n8_val:.1f}%\nbest N=8',
            xy=(best_n8_bit + offset[2] * width, best_n8_val),
            xytext=(best_n8_bit + offset[2] * width + 0.5, best_n8_val + 3),
            fontsize=8, color=colors['n8'],
            arrowprops=dict(arrowstyle='->', color=colors['n8'], lw=1),
            ha='center')

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel('Bit position', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('SimpleCNN per-bit accuracy: training vs attack dataset',
             fontsize=12, fontweight='normal', pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f'Bit {b}' for b in bits], fontsize=10)
ax.set_ylim(40, 82)
ax.set_yticks(range(40, 85, 5))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(fontsize=9, framealpha=0.4, loc='upper right',
          ncol=2, columnspacing=1.0)

# ── Summary stats below plot ──────────────────────────────────────────────────
mean_n1  = np.mean(atk_n1)
mean_n8  = np.mean(atk_n8)

fig.text(0.13, 0.01,
         f'Mean N=1: {mean_n1:.2f}%   |   '
         f'Mean N=8: {mean_n8:.2f}%   |   '
         f'Best N=1: Bit {best_n1_bit} ({best_n1_val:.1f}%)   |   '
         f'Best N=8: Bit {best_n8_bit} ({best_n8_val:.1f}%)',
         fontsize=8, color='#888780')

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('single_bit_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('single_bit_comparison.png', bbox_inches='tight', dpi=300)
print('[+] Saved: single_bit_comparison.pdf and single_bit_comparison.png')
plt.show()
