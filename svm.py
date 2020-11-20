import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dcase import get_features_filename, get_training_data, get_test_data, load_DCASE_development, load_DCASE_evaluation, features_extraction_DCASE, get_k_fold


def save_model(clf, layer):
    filename = "svm" + "_".join([str(key)+"="+str(val) for key, val in clf.params.items()]) + "_{}" + ".pck"
    model_path = os.path.join("model", filename)
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
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    parameters = [
        {"svc__C": np.linspace(1e-6,1e2,num=20)},
    ]
    clf = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=4, refit=True, verbose=2)
    
    t0 = time()
    print("Fitting the model...")
    clf.fit(X, y)
    print(time()-t0, " seconds")
    print(clf.cv_results_)
    print("Best params : ", clf.best_params_)
    print("Best score : ", clf.best_score_)
    save_model(clf, 4)
    return clf

def evaluating(db, clf, layer):
    from time import time
    print("Loading test data...")
    t0 = time()
    X_test, y_test = get_test_data(db, layer, n_jobs=2)
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
    return scores.mean()
    

if __name__ == "__main__":
    db = load_DCASE_development()
    features_extraction_DCASE(db)
    clf = training(db, 4)
    db_eval = load_DCASE_evaluation()
    features_extraction_DCASE(db_eval)
    acc = evaluating(db_eval, clf, 4)
    print("Evaluation accuracy : ", )
