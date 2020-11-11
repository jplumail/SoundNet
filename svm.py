import wave
import numpy as np
import torch
from soundnet import SoundNet
from sklearn.svm import LinearSVC
from time import time
import os
import matplotlib.pyplot as plt


def features_extraction(sound, layer):
    s = np.array(list(sound), dtype=np.float32) * (2 / 255) - 1
    s = torch.as_tensor(s)
    s = s.view((1, 1, -1, 1))
    with torch.no_grad():
        features = model.forward(s)[layer]
    return features.numpy().reshape(-1)


def get_training_data(setup_path, layer):
    data_path = ".".join(setup_path.split(".")[:-1]) + ".npz"
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


model = SoundNet()
model.load_state_dict(torch.load("sound8.pth"))

evaluation_setup_path = "DCASE/development/"
X,y = get_training_data(evaluation_setup_path + "meta.txt", 6)
