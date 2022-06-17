import itertools
import matplotlib.pyplot as plt
import numpy as np


classes = ['Hand Open', 'Hand Close', 'Wrist Flexion', 'Wrist Extension', 'Wrist Supination', 
           'Wrist Pronation', 'Ulnar Flexion', 'Radial Flexion', 'Tripod', 'Lateral Tripod', 'Coin Pinch', 'Precision Disk', 'Light Tool', 'Thumb Finger Extension',
           'Index Finger Extension', 'Medium Wrap', 'Power Sphere', 'Lateral Pinch', 'Victory', 'Good', 'Calling', 'OK']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig, ax1 = plt.subplots(figsize=(20, 20), ncols=1)
    plt.title(title, fontsize=20)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    pos = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(pos)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.show()