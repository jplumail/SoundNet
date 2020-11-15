import numpy as np
import torch
from soundnet import SoundNet
from util import preprocess, load_audio
from time import time
import os
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
import pickle


def load_DCASE_development():
    if not os.path.exists("DCASE"):
        os.mkdir("DCASE")
    db = TUTAcousticScenes_2017_DevelopmentSet(data_path="DCASE", filelisthash_exclude_dirs="features")
    db.initialize()
    return db

def load_DCASE_evaluation():
    if not os.path.exists("DCASE"):
        os.mkdir("DCASE")
    db = TUTAcousticScenes_2017_EvaluationSet(data_path="DCASE", filelisthash_exclude_dirs="features")
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


def get_k_fold(db):
    for fold in db.folds():
        i = 0
        train, evaluation = [], []
        for label in db.scene_labels():
            for item in db.train(fold=fold).filter(scene_label=label):
                train.append(i)
                i += 1
        for label in db.scene_labels():
            for item in db.eval(fold=fold).filter(scene_label=label):
                evaluation.append(i)
                i += 1
        yield np.array(train), np.array(evaluation)

def get_training_data(db, layer):
    X, y = [], []
    features_dir = os.path.join(db.local_path, "features")
    for fold in db.folds():
        for label in db.scene_labels():
            for item in db.train(fold=fold).filter(scene_label=label):
                features_filename = get_features_filename(features_dir, item.filename)
                x = np.load(features_filename)["layer"+str(layer)]
                X.append(x)
                y.append(label)
        for label in db.scene_labels():
            for item in db.eval(fold=fold).filter(scene_label=label):
                features_filename = get_features_filename(features_dir, item.filename)
                x = np.load(features_filename)["layer"+str(layer)]
                X.append(x)
                y.append(label)
    return np.array(X), np.array(y)

def get_test_data(db, layer):
    X, y = [], []
    features_dir = os.path.join(db.local_path, "features")
    for label in db.scene_labels():
        print(label)
        for item in db.test().filter(scene_label=label):
            features_filename = get_features_filename(features_dir, item.filename)
            x = np.load(features_filename)["layer"+str(layer)]
            X.append(x)
            y.append(label)
    return np.array(X), np.array(y)

def training(db, layer):
    from time import time
    t1 = time()
    print("Loading training data...")
    X, y = get_training_data(db, layer)
    print(time()-t1, " seconds")

    cv = get_k_fold(db)
    pipeline = make_pipeline(StandardScaler(), SVC())
    parameters = [
        {"svc__C": np.linspace(1e-3,1e-2,num=10), "svc__kernel": ["linear"]},
    ]
    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, refit=True, verbose=2)
    
    t0 = time()
    print("Fitting the model...")
    clf.fit(X, y)
    print(time()-t0, " seconds")
    print(clf.cv_results_)
    print("Best params : ", clf.best_params_)
    print("Best score : ", clf.best_score_)
    with open("model4.pk", "wb") as f:
        pickle.dump(clf.best_estimator_, f)
    return clf

def evaluating(db, clf, layer):
    from time import time
    print("Loading test data...")
    t0 = time()
    X_test, y_test = get_test_data(db, layer)
    print(time()-t0)
    scores = []
    n = 10
    N = len(X_test)
    batch = N//n
    for i in range(n-1):
        print(i)
        X, y = X_test[i*batch:(i+1)*batch], y_test[i*batch:(i+1)*batch]
        scores.append(clf.score(X,y))
    if n*batch < N:
        X, y = X_test[n*batch:], y_test[n*batch:]
        scores.append(clf.score(X,y))
    scores = np.array(scores)
    print("Accuracy : ", scores.mean())
    return scores.mean()
    

if __name__ == "__main__":
    db = load_DCASE_evaluation()
    features_extraction_DCASE(db)
    with open("model4.pk", "rb") as f:
        clf = pickle.load(f)
    evaluating(db, clf, 4)
