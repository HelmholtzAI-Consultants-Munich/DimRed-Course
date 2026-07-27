"""Generate the concept diagrams used in the method pages.

All figures are drawn from synthetic data or from a standard bundled dataset
(scikit-learn's ``digits``), so they carry no third-party image copyright and can
be freely regenerated and restyled.

Requires: numpy, scipy, matplotlib, scikit-learn and umap-learn.
Run:  python generate_concept_diagrams.py
It writes pca_concept.png, ica_concept.png, digits_comparison.png, nmds_shepard.png,
feature_clustering_concept.png and elastic_net_objective.png next to this script.
"""
import os
import warnings
from itertools import permutations

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.isotonic import IsotonicRegression
import umap

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

# ---------------- PCA vs t-SNE vs UMAP (same data, three embeddings) ----------------
digits = load_digits()
Xd, yd = digits.data, digits.target          # 1797 x 64, ten classes 0-9
emb_pca = PCA(n_components=2).fit_transform(Xd)
emb_tsne = TSNE(n_components=2, init="pca", perplexity=30,
                random_state=0).fit_transform(Xd)
emb_umap = umap.UMAP(n_components=2, random_state=0).fit_transform(Xd)

panels = [("PCA (linear)", emb_pca), ("t-SNE", emb_tsne), ("UMAP", emb_umap)]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True)
for ax, (name, emb) in zip(axes, panels):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=yd, cmap="tab10", s=8, alpha=0.8,
                    edgecolor="none")
    ax.set_title(name, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("The same data (handwritten digits, 0-9) embedded three ways", fontsize=13)
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), ticks=range(10),
                    fraction=0.02, pad=0.01)
cbar.set_label("digit class")
plt.savefig(os.path.join(HERE, "digits_comparison.png"), dpi=130,
            bbox_inches="tight", pad_inches=0.15); plt.close()

# ---------------- NMDS Shepard diagram (rank-order preservation) ----------------
Xr, _ = make_swiss_roll(n_samples=110, noise=0.15, random_state=0)
metric = MDS(n_components=2, metric=True, n_init=4, random_state=0,
             normalized_stress="auto").fit_transform(Xr)
# non-metric MDS, initialized from the metric solution so it does not collapse
emb_nmds = MDS(n_components=2, metric=False, n_init=1, max_iter=1000, random_state=0,
               normalized_stress="auto").fit_transform(Xr, init=metric)
d_orig, d_emb = pdist(Xr), pdist(emb_nmds)

fig, ax = plt.subplots(figsize=(5.4, 5))
ax.scatter(d_orig, d_emb, s=9, alpha=0.30, color="#3a86a8", edgecolor="none")
ir = IsotonicRegression().fit(d_orig, d_emb)
xs = np.linspace(d_orig.min(), d_orig.max(), 200)
ax.plot(xs, ir.predict(xs), color="#d1495b", lw=2.5,
        label="rank-order (monotonic) fit")
ax.set_xlabel("distance between two samples in the original space")
ax.set_ylabel("distance in the 2-D NMDS embedding")
ax.set_title("NMDS Shepard diagram: embedding distances rise\n"
             "monotonically with the original distances", fontsize=10.5)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.2)
plt.savefig(os.path.join(HERE, "nmds_shepard.png"), dpi=130,
            bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------- Data-driven feature clustering (feature aggregation) ----------------
rng = np.random.default_rng(3)
n = 200
groups = [7, 6, 5, 6]                        # feature-group sizes (24 features total)
cols = []
for size in groups:
    z = rng.normal(size=n)                   # one latent signal shared within the group
    for _ in range(size):
        cols.append(z * rng.uniform(0.75, 1.0) + 0.45 * rng.normal(size=n))
Ff = np.column_stack(cols)
Cf = np.corrcoef(Ff.T)

fig, ax = plt.subplots(figsize=(5.6, 5))
im = ax.imshow(Cf, cmap="RdBu_r", vmin=-1, vmax=1)
b = 0
for size in groups:                          # outline each discovered feature cluster
    ax.add_patch(plt.Rectangle((b - 0.5, b - 0.5), size, size, fill=False,
                               edgecolor="k", lw=2))
    b += size
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("features (e.g. genes)"); ax.set_ylabel("features (e.g. genes)")
ax.set_title("Data-driven aggregation: group correlated features,\n"
             "then summarize each group into one feature", fontsize=10.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("correlation between features")
plt.savefig(os.path.join(HERE, "feature_clustering_concept.png"), dpi=130,
            bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------- Elastic-net objective (rendered formula image) ----------------
# GitHub renders .rst without a math engine, so the formula is shipped as an image
# (as the course does for the ReliefF formula) to stay visible everywhere.
elastic_net = (r"$\min_{w,\,c}\ \ "
               r"C\sum_{i=1}^{n}\log\left(1+e^{-y_i(w^{T}x_i+c)}\right)"
               r"\ +\ \rho\,\|w\|_1"
               r"\ +\ \frac{1-\rho}{2}\,\|w\|_2^{2}$")
fig = plt.figure(figsize=(8.2, 1.25))
fig.text(0.5, 0.5, elastic_net, ha="center", va="center", fontsize=19)
fig.savefig(os.path.join(HERE, "elastic_net_objective.png"), dpi=200,
            bbox_inches="tight", pad_inches=0.2); plt.close()

print("saved pca_concept.png, ica_concept.png, digits_comparison.png, nmds_shepard.png, "
      "feature_clustering_concept.png, elastic_net_objective.png")
