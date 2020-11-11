import os
import wave
import numpy as np
import torch
from soundnet import SoundNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from pathlib import Path
from datetime import timedelta
from time import time


def features_extraction(sound, layer):
    s = np.array(list(sound), dtype=np.float32) * (2 / 255) - 1
    s = torch.as_tensor(s)
    s = s.view((1, 1, -1, 1))
    with torch.no_grad():
        features = model.forward(s)[layer]
    return features.numpy().reshape(-1)


def get_training_data(setup_path, layer):
    data_path = setup_path.parent / (setup_path.stem + ".npz")
    if data_path.exists():
        with data_path.open(mode="rb") as f:
            npzfile = np.load(f)
            X, y = npzfile["X"], npzfile["y"]
    else:
        X, y = [], []
        with setup_path.open() as f:
            N = 3507 if "train" in setup_path.stem else 1170
            t0 = time()
            for i, l in enumerate(f):
                wav_path, label = l.strip().split("\t")
                time_left = (N - i) * (time() - t0) / (i + 1)
                h, r = time_left // 3600, time_left % 3600
                m, s = r // 60, r % 60
                print(
                    "Extracting features {}... {}/{}({:2.2%}), {:n}h{:n}m{:.0f}s left  {}, {}".format(
                        setup_path, i, N, i / N, h, m, s, wav_path, label
                    ),
                    end="\r",
                )
                with wave.open("DCASE/development/" + wav_path) as sound:
                    s = sound.readframes(sound.getnframes())
                features = features_extraction(s, layer)
                X.append(features)
                y.append(label)
        X = np.array(X)
        with data_path.open(mode="wb") as f:
            npzfile = np.savez(f, X=X, y=y)
    return X, y


def get_test_data(setup_path, layer):
    X = []
    with open(setup_path, "r") as f:
        for l in f:
            wav_path = l.strip()
            print(wav_path)
            with wave.open("DCASE/development/" + wav_path) as sound:
                s = sound.readframes(sound.getnframes())
            features = features_extraction(s, layer)
            X.append(features)
    X = np.array(X)
    return X


def get_k_fold(k, layer):
    name = "fold{}_".format(k)
    train_path = evaluation_setup_path / (name + "train.txt")
    eval_path = evaluation_setup_path / (name + "evaluate.txt")
    X_train, y_train = get_training_data(train_path, layer)
    X_eval, y_eval = get_training_data(eval_path, layer)
    return X_train, y_train, X_eval, y_eval


model = SoundNet()
model.load_state_dict(torch.load("sound8.pth"))

evaluation_setup_path = Path("DCASE/development/evaluation_setup")
# X,y = get_training_data(evaluation_setup_path / "test.txt", 6)

for k in range(1, 5):
    X_train, y_train, X_eval, y_eval = get_k_fold(k, 6)
    clf = LinearSVC()
    # clf.fit(X_train, y_train)
    # print(clf.score(X_eval, y_eval))
