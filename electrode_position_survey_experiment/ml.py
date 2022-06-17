from matplotlib.pyplot import grid
from select_option_subject import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns
from time import time
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DATA_PATH = os.path.abspath(os.path.dirname(__file__))
SUB1_PATH = os.path.join(DATA_PATH, 'data', 'Test001')
SUB2_PATH = os.path.join(DATA_PATH, 'data', 'Test002')

def svc(subjectPath=SUB1_PATH, plan='A', show_results=False):
    # ============ load data =============
    data, label = BackPlanData(subjectPath, plan=plan)
    data, label = extract_feature(data, label)
    # ============ preprocessing =============
    std_scaler = StandardScaler()
    features = std_scaler.fit_transform(data)
    pca = PCA(n_components=0.99, whiten=True, svd_solver='auto')
    feature_pca = pca.fit_transform(features)
    # ============ set hyperparameters ============
    svm = SVC(kernel='rbf')
    gamma = np.logspace(-5, 1, 100)
    C = np.linspace(0.01, 10, 100)
    hyperparameters = dict(C=C, gamma=gamma)
    # ============ split the dataset ============
    kf = KFold(n_splits=5)
    acc_res = []
    f1_res = []
    for train_index, test_index in kf.split(feature_pca):
        Xtrain, Xtest = feature_pca[train_index], feature_pca[test_index]
        Ytrain, Ytest = label[train_index], label[test_index]

        # ============ train svm ================
        gridsearch = GridSearchCV(svm, hyperparameters, cv=10, verbose=2, n_jobs=-1)
        gridsearch.fit(Xtrain, Ytrain)
        # print(gridsearch.best_score_)
        # ============ choose the best estimator ============
        best_svm = gridsearch.best_estimator_
        test_score = best_svm.score(Xtest, Ytest)
        print("==================================")

        print(gridsearch.best_params_)

        print("==================================")

        # ============ show the result ==============
        if show_results:
            # ============ The effect of different parameter combinations on the results ==========
            results = pd.DataFrame(gridsearch.cv_results_)
            scores = np.array(results.mean_test_score).reshape(100, 100)
            sns.heatmap(scores, xticklabels=hyperparameters['gamma'], yticklabels=hyperparameters['C'])
            plt.show()

            # =========== confusion matrix =============
            conf_mx = confusion_matrix(Ytest, best_svm.predict(Xtest))
            plt.imshow(conf_mx, cmap=plt.cm.Blues)

            indices = range(len(conf_mx))

            plt.colorbar()

            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            #plt.title('Confusion Matrix')

            # 显示数据
            for first_index in range(len(conf_mx)):    #第几行
                for second_index in range(len(conf_mx[first_index])):    #第几列
                    #print(conf_mx[first_index][second_index])
                    plt.text(first_index, second_index, conf_mx.T[first_index][second_index])

            plt.show()

        res = classification_report(Ytest, best_svm.predict(Xtest), output_dict=True)
        f1_score_list = [res[str(i + 1)]['f1-score'] for i in range(4)]
        f1_score = sum(f1_score_list) / len(f1_score_list)

        acc_res.append(test_score)
        f1_res.append(f1_score)

    acc = sum(acc_res) / len(acc_res)
    f1_score = sum(f1_res) / len(f1_res)

    return acc, f1_score

def main():
    # ======================= subject 1, plan A ======================
    #              best parameter combinations 
    #              kernel = 'rbf', C = 4.651818181, gamma = 0.05722367
    #                          score = 0.91
    # ================================================================
    s1_A_result = svc(subjectPath=SUB1_PATH, plan='A', show_results=True)

    # ======================= subject 1, plan B ======================
    #              best parameter combinations 
    #              kernel = 'rbf', C = 7.67909, gamma = 0.0432876
    #                         score = 0.88
    # ================================================================
    s1_B_result = svc(subjectPath=SUB1_PATH, plan='B', show_results=False)
    

if __name__ == '__main__':
    main()