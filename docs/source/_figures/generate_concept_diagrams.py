"""Generate the PCA and ICA concept diagrams used in feature_transformation.rst.

The figures are drawn from synthetic data (no third-party images), so they can be
freely regenerated and restyled. Requires numpy, matplotlib and scikit-learn.

Run:  python generate_concept_diagrams.py
It writes pca_concept.png and ica_concept.png next to this script.
"""
import os
from itertools import permutations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- PCA concept ----------------
rng = np.random.default_rng(1)
X = rng.multivariate_normal([0, 0], [[3, 1.7], [1.7, 1.2]], 320)
mu = X.mean(0)
vals, vecs = np.linalg.eigh(np.cov((X - mu).T))
o = np.argsort(vals)[::-1]; vals, vecs = vals[o], vecs[:, o]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.45, color="#6699cc", edgecolor="none")
for k, (val, vec) in enumerate(zip(vals, vecs.T)):
    L = 2.6 * np.sqrt(val); col = ["#d1495b", "#edae49"][k]
    tip = mu + L * vec
    ax.annotate("", xy=tip, xytext=mu,
                arrowprops=dict(arrowstyle="-|>", lw=3, color=col))
    perp = np.array([-vec[1], vec[0]])
    if k == 0:                                  # PC1: beside the middle of the shaft
        pos = mu + 0.55 * L * vec + 0.75 * perp
    else:                                       # PC2: just beyond the short arrow tip
        pos = tip + 0.5 * vec
    ax.text(pos[0], pos[1], f"PC{k+1}", color=col, fontsize=13,
            fontweight="bold", ha="center", va="center")
ax.set_aspect("equal"); ax.grid(alpha=0.2)
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_title("PCA: components are the directions of maximum variance", fontsize=10.5)
plt.savefig(os.path.join(HERE, "pca_concept.png"), dpi=130,
            bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------- ICA concept (source separation) ----------------
rng = np.random.default_rng(0)
n = 500; t = np.linspace(0, 8, n)
s1 = np.sin(2.2 * t)
s2 = np.sign(np.sin(3.1 * t))
S = np.c_[s1, s2] + 0.06 * rng.normal(size=(n, 2))
S = (S - S.mean(0)) / S.std(0)
A = np.array([[1.0, 0.7], [0.55, 1.0]])
Xmix = S @ A.T
Srec = FastICA(n_components=2, random_state=0, whiten="unit-variance").fit_transform(Xmix)

# align recovered sources to the originals (fix ICA's permutation + sign ambiguity)
best = None
for perm in permutations(range(2)):
    Sp = Srec[:, list(perm)]
    signs = np.array([np.sign(np.corrcoef(Sp[:, j], S[:, j])[0, 1]) for j in range(2)])
    Sp = Sp * signs
    err = sum(1 - abs(np.corrcoef(Sp[:, j], S[:, j])[0, 1]) for j in range(2))
    if best is None or err < best[0]:
        best = (err, Sp)
Srec = best[1]

fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
stages = ["Independent\nsources", "Observed\nmixtures", "ICA-recovered\nsources"]
data = [S, Xmix, Srec]
colors = ["#d1495b", "#3a86a8"]
for r in range(3):
    for c in range(2):
        ax = axes[r][c]
        ax.plot(t, data[r][:, c], color=colors[c], lw=1.1)
        ax.set_yticks([]); ax.margins(x=0.01)
        if r == 0:
            ax.set_title(f"Signal {c + 1}", fontsize=10.5)
    axes[r][0].set_ylabel(stages[r], fontsize=10.5)
for c in range(2):
    axes[2][c].set_xlabel("time", fontsize=10)
fig.suptitle("ICA: separating independent signals from their mixture", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(HERE, "ica_concept.png"), dpi=130,
            bbox_inches="tight", pad_inches=0.1); plt.close()

print("saved pca_concept.png and ica_concept.png")
