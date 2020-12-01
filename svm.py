import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report

from dcase import get_features_filename, get_training_data, get_test_data, load_DCASE_development, load_DCASE_evaluation, features_extraction_DCASE, get_k_fold


def save_model(clf, layer):
    filename = "layer=" + str(layer) + "_" + "-".join([str(key)+"="+str(val) for key, val in clf.best_params_.items()]) + "_{}" + ".pck"
    model_path = os.path.join("models", filename)
    i = 0
    while os.path.exists(model_path.format(i)):
        i += 1
    model_path = model_path.format(i)
    with open(model_path, "wb") as f:
        pickle.dump(clf.best_estimator_, f)
    return model_path

def training(db, layer):
    from time import time
    t1 = time()
    print("Loading training data...")
    X, y = get_training_data(db, layer, n_jobs=2)
    print(time()-t1, " seconds")

    cv = get_k_fold(db)
    pipeline = make_pipeline(StandardScaler(), SVC())
    parameters ={
        "svc__C": np.logspace(1.5, 2.5, num=3),
        "svc__gamma": np.logspace(-3.5, -2.5, num=3),
        "svc__kernel": ["rbf"]
    }
    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=4, refit=True, verbose=0)
    
    t0 = time()
    print("Fitting the model...")
    clf.fit(X, y)
    print("Fitting time : ", time()-t0, " seconds")

    print("Best parameters found : ")
    print(clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    
    save_model(clf, layer)
    
    return clf

def evaluating(db, clf, layer):
    from time import time
    print("Loading test data...")
    t0 = time()
    X_test, y_test = get_test_data(db, layer, n_jobs=2)
    print(time()-t0)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    #plot_confusion_matrix(clf, X_test, y_test)
    #plt.show()
    
    

if __name__ == "__main__":
    db = load_DCASE_development()
    features_extraction_DCASE(db)
    for layer in range(3,9):
        print("#############")
        print("LAYER : ", layer)
        clf = training(db, layer)
        db_eval = load_DCASE_evaluation()
        features_extraction_DCASE(db_eval)
        evaluating(db_eval, clf, layer)
