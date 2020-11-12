import wave
import numpy as np
import torch
from soundnet import SoundNet
from time import time
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier


def features_extraction(sound, layer):
    s = np.array(list(sound), dtype=np.float32) * (2 / 255) - 1
    s = torch.as_tensor(s)
    s = s.view((1, 1, -1, 1))
    with torch.no_grad():
        if layer > 6:
            features = model.forward(s)[7][layer - 7]
        else:
            features = model.forward(s)[layer]
    return features.numpy().reshape(-1)


def get_training_data(setup_path, layer):
    data_path = ".".join(setup_path.split(".")[:-1]) + str(layer) + ".npz"
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            npzfile = np.load(f)
            X, y = npzfile["X"], npzfile["y"]
    else:
        X, y = [], []
        with open(setup_path, "r") as f:
            N = 4680
            t0 = time()
            for i, l in enumerate(f):
                wav_path, label, _ = l.strip().split("\t")
                time_left = (N - i) * (time() - t0) / (i + 1)
                h, r = time_left // 3600, time_left % 3600
                m, s = r // 60, r % 60
                print(
                    "Extracting features {}... {}/{}({:2.2%}), {:n}h{:n}m{:.0f}s left  {}, {}".format(
                        setup_path, i, N, i / N, h, m, s, wav_path, label
                    ),
                    end="\r",
                )
                print("", end="\r")
                with wave.open("DCASE/development/" + wav_path) as sound:
                    s = sound.readframes(sound.getnframes())
                features = features_extraction(s, layer)
                X.append(features)
                y.append(label)
        X = np.array(X)
        with open(data_path, "wb") as f:
            npzfile = np.savez(f, X=X, y=y)
    return X, y


def k_fold(meta_path, setup_path, mode="training"):
    meta_txt = np.genfromtxt(meta_path, delimiter="\t", dtype="str")
    meta_files = [x for x in meta_txt[:, 0]]
    meta_short_files = [x for x in meta_txt[:, 2]]
    N = len(meta_files)
    file_indices = {}

    def get_meta_indices(files_list):
        indices = []
        for f in files_list:
            sound_file = f[6:10]
            if sound_file in file_indices:
                start_index = file_indices[sound_file]
            else:
                for i in range(N):
                    if sound_file == meta_short_files[i]:
                        start_index = i
                        file_indices[sound_file] = start_index
                        break
            for i in range(start_index, N):
                if f == meta_files[i]:
                    indices.append(i)
                    break
        return np.array(indices)

    for i in range(4):
        if mode == "training":
            train_path = setup_path + "fold{}_train.txt".format(i + 1)
            eval_path = setup_path + "fold{}_evaluate.txt".format(i + 1)
            train_txt = np.genfromtxt(train_path, delimiter="\t", dtype="str")
            eval_txt = np.genfromtxt(eval_path, delimiter="\t", dtype="str")
            train_files = [x for x in train_txt[:, 0]]
            eval_files = [x for x in eval_txt[:, 0]]
            yield get_meta_indices(train_files), get_meta_indices(eval_files)
        elif mode == "test":
            test_path = setup_path + "fold{}_test.txt".format(i + 1)
            test_txt = np.genfromtxt(test_path, delimiter="\t", dtype="str")
            test_files = [x for x in test_txt]
            yield get_meta_indices(test_files)
        else:
            print(mode + " mode not recognized")


model = SoundNet()
model.load_state_dict(torch.load("sound8.pth"))
"""
evaluation_setup_path = "DCASE/development/"
for layer in range(9):
    X, y = get_training_data(evaluation_setup_path + "meta.txt", layer)
"""

with open("DCASE/development/meta6.npz", "rb") as f:
    npzfile = np.load(f)
    X = npzfile["X"]
    y = npzfile["y"]

meta_path = "DCASE/development/meta.txt"
setup_path = "DCASE/development/evaluation_setup/"
splitter_training = k_fold(meta_path, setup_path)
splitter_testing = k_fold(meta_path, setup_path, mode="test")
clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1e5))

train0_ind = next(splitter_training)[0]
X_train, y_train = X[train0_ind], y[train0_ind]
clf.fit(X_train[100:], y_train[100:])
print("Score training : ", clf.score(X_train[100:], y_train[100:]))

test0_ind = next(splitter_testing)
X_test, y_test = X[test0_ind], y[test0_ind]
print("Score testing : ",clf.score(X_test, y_test))