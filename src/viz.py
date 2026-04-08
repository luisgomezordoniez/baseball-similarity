import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.model import FEATURES_DEFAULT as FEATURES

COLORS = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

def radar_chart(labels: list, raw_df: pd.DataFrame):
    """Generate a radar chart comparing one or more player-seasons."""
    normed = raw_df.copy()
    for col in FEATURES:
        col_min = raw_df[col].min()
        col_max = raw_df[col].max()
        normed[col] = (raw_df[col] - col_min) / (col_max - col_min + 1e-9)
    normed = normed.set_index('Label')

    n = len(FEATURES)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('none')

    for i, label in enumerate(labels):
        if label not in normed.index:
            continue
        vals = normed.loc[label, FEATURES].tolist()
        vals += [vals[0]]  # close the polygon
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURES, fontsize=11)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
    plt.tight_layout()
    return fig