from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from show_confusion import *
from sklearn.model_selection import GridSearchCV
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

def train_knn(X, y, show_confusion=False):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, 
                                                random_state=42)
    knn = KNeighborsClassifier()

    n_neighbors = np.arange(1, 50, 1)
    weights = ['uniform', 'distance']
    p = np.arange(1, 10, 1)
    
    hyperparameters = dict( 
                            n_neighbors=n_neighbors,
                            weights=weights, 
                            p=p,
                            )

    gridsearch = GridSearchCV(knn, hyperparameters, cv=10, verbose=2, n_jobs=-1)
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
    f1_score_list = [res[str(i+1)]['f1-score'] for i in range(22)]
    f1_score = sum(f1_score_list) / len(f1_score_list)

    return acc, f1_score

def fusion_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fusion_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_knn(X, y)
    return acc, f1

def semg_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'semg_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_knn(X, y)
    return acc, f1

def fmg_model():
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fmg_feature.csv'))
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = np.array(data.copy().drop(['label'], axis=1))
    y = data['label'].copy()
    acc, f1 = train_knn(X, y)
    return acc, f1

if __name__ == '__main__':
    # fusion model (sEMG + FMG)
    # acc, f1 = fusion_model()
    # semg model
    acc, f1 = fmg_model()
    print(acc)
    print(f1)