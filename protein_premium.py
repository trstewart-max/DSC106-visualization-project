import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 10, 'legend.fontsize': 11,
})

df = pd.read_csv('/mnt/user-data/uploads/grocerydb.csv')
df = df[df['FPro'].between(0, 1)]
df = df[df['price'] > 0]
df = df[df['price'] < df['price'].quantile(0.99)]
df = df[df['Protein'] > 0]
df['protein_per_dollar'] = df['Protein'] / df['price']
df = df[df['protein_per_dollar'] < df['protein_per_dollar'].quantile(0.99)]

LABELS = ['Minimal\nProcessing', 'Slightly\nProcessed',
          'Moderately\nProcessed', 'Highly\nProcessed']
df['fpro_bin'] = pd.cut(df['FPro'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=LABELS)

TARGET_COL  = '#CC0000'
WALMART_COL = '#0071CE'
WF_COL      = '#00674B'

data = {}
for store in ['Target', 'Walmart', 'WholeFoods']:
    g = (df[df['store'] == store]
         .groupby('fpro_bin', observed=True)
         .agg(ppd=('protein_per_dollar', 'median'))
         .reset_index())
    data[store] = g['ppd'].values

x   = np.arange(len(LABELS))
tgt = data['Target']
wmt = data['Walmart']
wf  = data['WholeFoods']

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ccc')
ax.spines['bottom'].set_color('#ccc')
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# Shaded gap
ax.fill_between(x, wf, np.maximum(tgt, wmt),
                alpha=0.10, color='grey', zorder=1)

# Lines
for vals, col, label, y_offset in [
    (tgt, TARGET_COL,  'Target',      0.10),
    (wmt, WALMART_COL, 'Walmart',    -0.12),
    (wf,  WF_COL,      'Whole Foods', 0.00),
]:
    ax.plot(x, vals, color=col, linewidth=3.0, zorder=4,
            marker='o', markersize=9,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.text(x[-1] + 0.10, vals[-1] + y_offset, label,
            color=col, fontsize=12, fontweight='bold', va='center')

# ── Vertical bracket arrows ───────────────────────────────────────────────────
# Arrow 1: at x=0 (Fresh), Target vs WF
gap_fresh = tgt[0] / wf[0]
ax.annotate('', xy=(0, wf[0]), xytext=(0, tgt[0]),
            arrowprops=dict(arrowstyle='<->', color='#DAA520',
                            lw=2.2, mutation_scale=14))
ax.text(0.12, (tgt[0] + wf[0]) / 2,
        f'~{gap_fresh:.1f}× more protein\nper dollar',
        fontsize=9.5, color='#9A7000', va='center',
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#DAA520', alpha=0.95))

# Arrow 2: at x=3 (Highly Processed), Target vs WF
gap_proc = tgt[-1] / wf[-1]
ax.annotate('', xy=(3, wf[-1]), xytext=(3, tgt[-1]),
            arrowprops=dict(arrowstyle='<->', color='#DAA520',
                            lw=2.2, mutation_scale=14))
ax.text(2.55, (tgt[-1] + wf[-1]) / 2,
        f'~{gap_proc:.1f}× less protein\nper dollar',
        fontsize=9.5, color='#9A7000', va='center', ha='right',
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#DAA520', alpha=0.95))

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_xlabel('Food Processing Level', labelpad=12)
ax.set_ylabel('Protein Per Dollar', labelpad=12)
ax.set_xlim(-0.3, 3.75)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

ax.set_title(
    "Whole Foods Isn't Built for Budget Lifters —\nEven at Higher Quality",
    pad=14, loc='center',
)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/18_protein_premium.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Done.")
