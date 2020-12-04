import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
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
    t1 = time()
    print("Loading training data...")
    X, y, _ = get_training_data(db, layer, n_jobs=4)
    print(time()-t1, " seconds")

    cv = get_k_fold(db)
    pipeline = make_pipeline(StandardScaler(), SVC())
    C_range = np.logspace(-2, 7, num=5)
    gamma_range = np.logspace(-7, -2, num=5)
    parameters ={
        "svc__C": C_range,
        "svc__gamma": gamma_range,
    }
    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=4, refit=True, verbose=0)
    
    t0 = time()
    print("Fitting the model...")
    clf.fit(X, y)
    print("Fitting time : ", time()-t0, " seconds")

    print("Best parameters found : ")
    print(clf.best_params_)

    scores = clf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), ["%.2e"%f for f in gamma_range], rotation=45)
    plt.yticks(np.arange(len(C_range)), ["%.2e"%f for f in C_range])
    plt.title('Validation accuracy')
    plt.savefig("training-layer={}.png".format(layer), format="png")
    save_model(clf,layer)

    return clf

def evaluating(db, clf, layer):
    print("Loading test data...")
    t0 = time()
    X_test, y_test, _ = get_test_data(db, layer, n_jobs=4)
    print(time()-t0)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    #plot_confusion_matrix(clf, X_test, y_test)
    #plt.show()
    
    

if __name__ == "__main__":
    db = load_DCASE_development()
    features_extraction_DCASE(db)
    for layer in range(3, 9):
        print("#############")
        print("LAYER : ", layer)
        clf = training(db, layer)
        db_eval = load_DCASE_evaluation()
        features_extraction_DCASE(db_eval)
        evaluating(db_eval, clf, layer)
