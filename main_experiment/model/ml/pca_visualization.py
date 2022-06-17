import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fusion_feature.csv'))


def scatterPlot(xDF, yDF, algoName):
    
    xDF = pd.DataFrame(data=xDF, index=range(xDF.shape[0]))
    yDF = pd.DataFrame(data=yDF, index=range(yDF.shape[0]))
    
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    
    #plt.figure(figsize=(10,10))
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", \
               data=tempDF, fit_reg=False, size=6.5)

    ax = plt.gca()
    ax.set_title("Separation of Observations using "+algoName)
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH, index_col=0)
    dataX = np.array(data.copy().drop(['label'], axis=1))
    dataY = data['label'].copy()
    scatterPlot(dataX, dataY, "PCA")