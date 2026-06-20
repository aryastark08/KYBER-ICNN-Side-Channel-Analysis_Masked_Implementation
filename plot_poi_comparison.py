import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────────────────────
correlations = np.load('correlation_profile.npy')
cnn_pois     = np.load('cnn_optimal_pois.npy')   # (32, 534)

# CNN POIs for byte 0 only (for visualization)
cnn_pois_byte0 = cnn_pois[0]

# ── LDA POI regions (from researcher formula, i=0) ────────────────────────────
lda_range1 = list(range(233, 634))    # 401 points
lda_range2 = list(range(3367, 3500))  # 133 points
lda_all    = lda_range1 + lda_range2  # 534 total

# ── CNN POIs outside LDA regions ─────────────────────────────────────────────
cnn_outside_lda = [t for t in cnn_pois_byte0
                   if t not in lda_range1 and t not in lda_range2]
cnn_inside_lda  = [t for t in cnn_pois_byte0
                   if t in lda_range1 or t in lda_range2]

# ── Find top 10 correlated points ─────────────────────────────────────────────
top10_idx  = np.argsort(correlations)[-10:][::-1]
top10_corr = correlations[top10_idx]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150,
                         gridspec_kw={'height_ratios': [3, 1]})

# ── TOP PANEL: Full correlation profile ───────────────────────────────────────
ax = axes[0]

time_points = np.arange(len(correlations))

# Full correlation line
ax.plot(time_points, correlations,
        color='#B4B2A9', linewidth=0.6, alpha=0.8, zorder=2,
        label='Pearson correlation $\\rho_t$')

# Overall mean line
mean_corr = correlations.mean()
ax.axhline(y=mean_corr, color='#888780', linewidth=1,
           linestyle=':', zorder=3,
           label=f'Overall mean ($\\rho$ = {mean_corr:.4f})')

# LDA POI region 1 shading
ax.axvspan(233, 633, alpha=0.15, color='#3266AD', zorder=1,
           label='LDA POI region 1 (t=233–633)')

# LDA POI region 2 shading
ax.axvspan(3367, 3499, alpha=0.15, color='#3266AD', zorder=1,
           label='LDA POI region 2 (t=3367–3499)')

# CNN POIs outside LDA (red dots)
ax.scatter(cnn_outside_lda,
           correlations[cnn_outside_lda],
           color='#D85A30', s=6, zorder=4, alpha=0.7,
           label=f'CNN POIs outside LDA regions ({len(cnn_outside_lda)} points)')

# CNN POIs inside LDA (green dots)
ax.scatter(cnn_inside_lda,
           correlations[cnn_inside_lda],
           color='#1D9E75', s=6, zorder=4, alpha=0.7,
           label=f'CNN POIs inside LDA regions ({len(cnn_inside_lda)} points)')

# Annotate ONLY the single strongest point (t=5944)
best_idx  = top10_idx[0]
best_corr = top10_corr[0]
in_lda    = best_idx in lda_all
color     = '#1D9E75' if in_lda else '#D85A30'
label     = 'inside LDA' if in_lda else 'outside LDA'

ax.annotate(f't={best_idx},  $\\rho$={best_corr:.3f}  ({label})',
            xy=(best_idx, best_corr),
            xytext=(best_idx - 900, best_corr + 0.008),
            fontsize=9, color=color, fontweight='normal',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
            ha='left', zorder=5)

ax.set_xlim(0, len(correlations))
ax.set_ylim(0, correlations.max() * 1.25)
ax.set_ylabel('Pearson correlation |$\\rho_t$|', fontsize=11)
ax.set_title('Correlation profile across 6,000 time points: LDA POIs vs CNN-optimized POIs',
             fontsize=11, fontweight='normal', pad=10)
ax.legend(fontsize=8, framealpha=0.4, loc='upper left',
          ncol=2, columnspacing=1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2)
ax.tick_params(axis='x', labelbottom=False)

# ── BOTTOM PANEL: Where are the POIs? ────────────────────────────────────────
ax2 = axes[1]

# LDA POI positions
ax2.scatter(lda_all, [0.6] * len(lda_all),
            color='#3266AD', s=4, alpha=0.5, zorder=2,
            label='LDA POIs')

# CNN POIs outside LDA
ax2.scatter(cnn_outside_lda, [0.3] * len(cnn_outside_lda),
            color='#D85A30', s=4, alpha=0.5, zorder=2,
            label='CNN POIs (outside LDA)')

# CNN POIs inside LDA
ax2.scatter(cnn_inside_lda, [0.3] * len(cnn_inside_lda),
            color='#1D9E75', s=4, alpha=0.5, zorder=2,
            label='CNN POIs (inside LDA)')

ax2.set_xlim(0, len(correlations))
ax2.set_ylim(0, 1)
ax2.set_yticks([0.3, 0.6])
ax2.set_yticklabels(['CNN POIs', 'LDA POIs'], fontsize=9)
ax2.set_xlabel('Time sample index (0–5,999)', fontsize=11)
ax2.legend(fontsize=8, framealpha=0.4, loc='upper right', ncol=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.2)

# ── Stats box ─────────────────────────────────────────────────────────────────
lda_mean = correlations[lda_all].mean()
cnn_mean = correlations[cnn_pois_byte0].mean()
overlap  = len(cnn_inside_lda)
outside  = len(cnn_outside_lda)

stats_text = (
    f'LDA POI mean correlation : {lda_mean:.4f}\n'
    f'CNN POI mean correlation : {cnn_mean:.4f}\n'
    f'CNN POIs inside LDA      : {overlap} / 534\n'
    f'CNN POIs outside LDA     : {outside} / 534\n'
    f'Strongest point          : t={top10_idx[0]} ($\\rho$={top10_corr[0]:.4f})'
)

fig.text(0.72, 0.30, stats_text,
         fontsize=8.5,
         color='#444441',
         bbox=dict(boxstyle='round,pad=0.5',
                   facecolor='#F1EFE8',
                   edgecolor='#D3D1C7',
                   alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.08)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('poi_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('poi_comparison.png', bbox_inches='tight', dpi=300)
print('[+] Saved: poi_comparison.pdf and poi_comparison.png')
plt.show()