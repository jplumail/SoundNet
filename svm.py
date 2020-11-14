import numpy as np
import torch
from soundnet import SoundNet
from util import preprocess, load_audio
from time import time
import os
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet
from sklearn.svm import LinearSVC, SVC
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

def features_extraction_DCASE(db):
    features_dir = os.path.join(db.local_path, "features")
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)
    model = SoundNet()
    model.load_state_dict(torch.load("sound8.pth"))
    model.eval()
    for audio_filename in db.audio_files:
        features_filename = get_features_filename(features_dir, audio_filename)
        if not os.path.exists(features_filename):
            x = extract_features(audio_filename, model)
            save_features(x, features_filename)

def get_features_filename(features_dir, audio_filename):
    parent, last = os.path.split(audio_filename)
    features_filename_last = last.replace(".wav", ".npz")
    features_filename = os.path.join(features_dir, features_filename_last)
    return features_filename

def save_features(x, feature_filename):
    features_name = {"layer"+str(i) : x[i] for i in range(len(x))}
    np.savez(feature_filename, **features_name)

def extract_features(audio_filename, model):
    sound, sr = load_audio(audio_filename, sr=44100)
    sound = preprocess(sound, config={"load_size": 44100*10, "phase": "extract"})
    sound = torch.as_tensor(sound)
    with torch.no_grad():
        features = model.forward(sound)
    features = features[:7] + [features[7][0], features[7][1]]
    features = [f.numpy().reshape(-1) for f in features]
    return features


def training(db, layer):
    clf = SVC(C=10)
    features_dir = os.path.join(db.local_path, "features")
    fold = 1
    X, y = [], []
    t0 = time()
    for label in db.scene_labels():
        print(label)
        for item in db.train(fold=fold).filter(scene_label=label):
            features_filename = get_features_filename(features_dir, item.filename)
            x = np.load(features_filename)["layer"+str(layer)]
            X.append(x)
            y.append(label)
    X = np.array(X)
    t0 = time()
    scaler = StandardScaler(copy=False)
    X_new = scaler.fit_transform(X)
    clf.fit(X_new, y)
    print("Training accuracy : ", clf.score(X_new, y))
    del X_new
    X, y = [], []
    for label in db.scene_labels():
        for item in db.eval(fold=fold).filter(scene_label=label):
            features_filename = get_features_filename(features_dir, item.filename)
            x = np.load(features_filename)["layer"+str(layer)]
            X.append(x)
            y.append(label)
    print("Testing accuracy : ", clf.score(scaler.transform(X),y))

db = load_DCASE()
features_extraction_DCASE(db)
training(db, 5)