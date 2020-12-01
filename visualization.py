from dcase import load_DCASE_development, get_features_filename, get_training_data
import numpy as np
import os
from time import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap


def draw_umap(layer):
    db = load_DCASE_development()
    t0 = time()
    print("Loading training data...")
    X, y = get_training_data(db, layer, n_jobs=4)
    print("in ", time()-t0)
    print(X.shape)

    print("Sclaling...")
    scaled_X = StandardScaler().fit_transform(X)

    print("Fitting uMap...")
    reducer = umap.UMAP(n_neighbors=1000)
    embedding = reducer.fit_transform(scaled_X)

    print("Plotting...")
    labels = np.unique(np.array(y))
    colors = plt.cm.rainbow(np.linspace(0, 1, num=len(labels)))
    dic = {l: c for l,c in zip(labels, colors)}
    c = [dic[l] for l in y]
    fig, ax = plt.subplots(figsize=(10,10))
    for l in labels:
        e = embedding[y==l]
        ax.scatter(
            e[:, 0],
            e[:, 1],
            cmap=dic[l],
            label=l,
            s=30,
        )
    fig.gca().set_aspect('equal', 'datalim')
    ax.legend()

    return fig, ax

if __name__ == "__main__":
    draw_umap(1)
    plt.show()