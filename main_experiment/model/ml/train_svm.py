from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
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

def train_svm(X, y, show_confusion=False):
    kf = KFold(n_splits=5)
    acc_res = []
    f1_res = []
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = y[train_index], y[test_index]
        svm = SVC(kernel='rbf')
        gamma = np.logspace(-5, 1, 1000)
        C = np.linspace(0.01, 10, 1000)
        hyperparameters = dict(C=C, gamma=gamma)
        gridsearch = GridSearchCV(svm, hyperparameters, cv=10, verbose=2, n_jobs=-1)
        gridsearch.fit(Xtrain, Ytrain)
        best_svm = gridsearch.best_estimator_
        print("Best parameters")
        print(gridsearch.best_params_)
        acc = best_svm.score(Xtest, Ytest)

        if show_confusion:
            conf_mx = confusion_matrix(Ytest, best_svm.predict(Xtest))
            plot_confusion_matrix(conf_mx, classes=classes, normalize=True,
                        title='Confusion matrix, with normalization')   

        res = classification_report(Ytest, best_svm.predict(Xtest), output_dict=True)
        # print(res)
        f1_score_list = [res[str(i+1)]['f1-score'] for i in range(22)]
        f1_score = sum(f1_score_list) / len(f1_score_list)

        acc_res.append(acc)
        f1_res.append(f1_score)

    acc = sum(acc_res) / len(acc_res)
    f1_score = sum(f1_res) / len(f1_res)

    return acc, f1_score

def fusion_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fusion_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_svm(X, y)
    return acc, f1

def semg_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'semg_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_svm(X, y)
    return acc, f1

def fmg_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fmg_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_svm(X, y)
    return acc, f1

if __name__ == '__main__':
    # fusion model (sEMG + FMG)
    # acc, f1 = fusion_model()
    # semg model
    acc, f1 = fmg_model()
    print(acc)
    print(f1)