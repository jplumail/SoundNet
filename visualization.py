from dcase import load_DCASE_development, get_features_filename
import numpy as np
import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap


def draw_umap(layer, avoidOverloadingMemory=True)

    db = load_DCASE_development()
    layer = 1

    X = []
    y = []
    features_dir = os.path.join(db.local_path, "features")
    for k in db.folds():
        for label in db.scene_labels():
            print(label)
            for item in db.train(fold=k).filter(scene_label=label):
                features_filename = get_features_filename(features_dir, item.filename)
                try:
                    x = np.load(features_filename)["layer"+str(layer)]
                    X.append(x.reshape(-1))
                    y.append(label)
                except FileNotFoundError:
                    pass
        if avoidOverloadingMemory:
            break
    X = np.array(X)
    y = np.array(y)
    scaled_X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_neighbors=100)
    embedding = reducer.fit_transform(scaled_X)

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