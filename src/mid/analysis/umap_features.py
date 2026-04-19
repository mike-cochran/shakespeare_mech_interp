"""UMAP projection of the SAE decoder dictionary.

Owner: Areeb
"""

from __future__ import annotations

import numpy as np


def project_features(
    sae,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UMAP on ``sae.W_dec`` (shape ``[n_features, d_model]``).

    Returns ``(embeddings, feature_indices)`` where ``embeddings`` is
    ``(n_features, n_components)`` and ``feature_indices`` is
    ``np.arange(n_features)``. Cosine metric is used because decoder
    directions are what matter, not their magnitudes.
    """
    import umap

    W_dec = sae.W_dec.detach().cpu().numpy()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embeddings = reducer.fit_transform(W_dec)
    feature_indices = np.arange(W_dec.shape[0])
    return embeddings, feature_indices


def plot_feature_map(
    embeddings_2d: np.ndarray,
    labels: dict[int, str] | None = None,
    highlight_idxs: list[int] | None = None,
    save_path: str | None = None,
    title: str = "SAE feature map (UMAP of decoder directions)",
):
    """Scatter plot of 2D-projected features.

    If ``labels={feature_idx: label_str}`` is given, points are colored
    by label. ``highlight_idxs`` draws red circles + annotations on
    specific features. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    xs, ys = embeddings_2d[:, 0], embeddings_2d[:, 1]

    if labels is None:
        ax.scatter(xs, ys, s=6, alpha=0.5, c="steelblue")
    else:
        unique_labels = sorted(set(labels.values()))
        label_to_color = {lbl: plt.cm.tab20(i % 20) for i, lbl in enumerate(unique_labels)}
        colors = [
            label_to_color[labels[i]] if i in labels else (0.7, 0.7, 0.7, 0.3)
            for i in range(len(embeddings_2d))
        ]
        ax.scatter(xs, ys, s=8, alpha=0.6, c=colors)

    if highlight_idxs:
        for idx in highlight_idxs:
            ax.scatter(
                xs[idx],
                ys[idx],
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
            )
            annotation = labels[idx] if labels and idx in labels else f"feat {idx}"
            ax.annotate(
                annotation,
                (xs[idx], ys[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="red",
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
