import wave
import numpy as np
import torch
from soundnet import SoundNet
from time import time
import os
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier


def load_DCASE():
    if not os.path.exists("DCASE"):
        os.mkdir("DCASE")
    db = TUTAcousticScenes_2017_DevelopmentSet(data_path="DCASE", filelisthash_exclude_dirs="features")
    db.initialize()
    return db

def features_extraction_DCASE():
    db = TUTAcousticScenes_2017_DevelopmentSet(data_path="DCASE", filelisthash_exclude_dirs="features") # TO REMOVE
    #db = load_DCASE()
    features_dir = os.path.join(db.local_path, "features")
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)
    model = SoundNet()
    model.load_state_dict(torch.load("sound8.pth"))
    for audio_filename in db.audio_files:
        parent, last = os.path.split(audio_filename)
        features_filename_last = last.replace(".wav", ".npz")
        features_filename = os.path.join(features_dir, features_filename_last)
        if not os.path.exists(features_filename):
            x = extract_features(audio_filename, model)
            save_features(x, features_filename)

def save_features(x, feature_filename):
    features_name = {"layer"+str(i) : x[i] for i in range(len(x))}
    np.savez(feature_filename, **features_name)

def extract_features(audio_filename, model):
    with wave.open(audio_filename) as sound:
        s = sound.readframes(sound.getnframes())
    s = np.array(list(s), dtype=np.float32) * 2 - 256
    s = torch.as_tensor(s)
    s = s.view((1, 1, -1, 1))
    with torch.no_grad():
        features = model.forward(s)
    features = features[:7] + [features[7][0], features[7][1]]
    return [f.numpy().reshape(-1) for f in features]

features_extraction_DCASE()


"""
evaluation_setup_path = "DCASE/development/"
for layer in range(9):
    X, y = get_training_data(evaluation_setup_path + "meta.txt", layer)
"""



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
"""