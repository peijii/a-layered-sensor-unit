from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import xgboost as xgb
from show_confusion import *
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

classes = ['Hand Open', 'Hand Close', 'Wrist Flexion', 'Wrist Extension', 'Wrist Supination', 
           'Wrist Pronation', 'Ulnar Flexion', 'Radial Flexion', 'Tripod', 'Lateral Tripod', 'Coin Pinch', 'Precision Disk', 'Light Tool', 'Thumb Finger Extension',
           'Index Finger Extension', 'Medium Wrap', 'Power Sphere', 'Lateral Pinch', 'Victory', 'Good', 'Calling', 'OK']


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fusion_feature.csv'))

def train_model(X, y, show_confusion=False):
    y -= 1
    kf = KFold(n_splits=5)
    acc_res = []
    f1_res = []
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = y[train_index], y[test_index]

        xgbc = xgb.XGBClassifier()

        n_estimators = np.arange(1, 300, 1)
        eta = np.linspace(0, 1, 100)
        max_depth = np.arange(1, 20, 1)

        hyperparameters = dict(
                                n_estimators=n_estimators,
                                eta=eta,
                                max_depth=max_depth
                                )

        gridsearch = GridSearchCV(xgbc, hyperparameters, cv=5, verbose=2, n_jobs=-1)
        gridsearch.fit(Xtrain, Ytrain)
        best_knn = gridsearch.best_estimator_
        print("Best parameters")
        print(gridsearch.best_params_)
        acc = best_knn.score(Xtest, Ytest)

        if show_confusion:
            conf_mx = confusion_matrix(Ytest, best_knn.predict(Xtest))
            plot_confusion_matrix(conf_mx, classes=classes, normalize=True,
                          title='Confusion matrix, with normalization')

        res = classification_report(Ytest, best_knn.predict(Xtest), output_dict=True)
        # print(res)
        f1_score_list = [res[str(i)]['f1-score'] for i in range(22)]
        f1_score = sum(f1_score_list) / len(f1_score_list)

        acc_res.append(acc)
        f1_res.append(f1_score)

    acc = sum(acc_res) / len(acc_res)
    f1_score = sum(f1_res) / len(f1_res)

    return acc, f1_score

def fusion_model(id):
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fusion_id.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    data = data[data['id']==id]
    X = np.array(data.copy().drop(['label', 'id'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_model(X, y)
    return acc, f1

def semg_model(id):
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'all_feature_id.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    data = data[data['id'] == id]
    semg_data = data.iloc[:, :30]
    label = data.iloc[:, -2]

    pca = PCA(n_components=10, whiten=True, svd_solver='auto')
    semg_data = pca.fit_transform(semg_data)

    X = np.array(semg_data)
    y = np.array(label)
    acc, f1 = train_model(X, y)
    return acc, f1

def fmg_model(id):
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'all_feature_id.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    data = data[data['id'] == id]
    fmg_data = data.iloc[:, 30:-2]
    label = data.iloc[:, -2]

    pca = PCA(n_components=19, whiten=True, svd_solver='auto')
    fmg_data = pca.fit_transform(fmg_data)

    X = np.array(fmg_data)
    y = np.array(label)
    acc, f1 = train_model(X, y)
    return acc, f1

if __name__ == '__main__':
    # fusion model (sEMG + FMG)
    # acc, f1 = fusion_model()
    # semg model
    res = []
    for i in range(9):
        sub = []
        acc, f1 = fusion_model(i+1)
        sub.append((str(i+1), 'fusion', acc, f1))
        
        acc, f1 = semg_model(i+1)
        sub.append((str(i+1), 'semg', acc, f1))

        acc, f1 = fmg_model(i+1)
        sub.append((str(i+1), 'fmg', acc, f1))

        res.append(sub)

    print(res)
    # acc, f1 = fusion_model(1)
    # print(acc)
    # print(f1)