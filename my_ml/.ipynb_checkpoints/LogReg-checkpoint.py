import pandas as pd
import numpy as np

import itertools

# Обучене модели и подготовка данных
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale, label_binarize
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# визуализация
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(12,8)})

import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(X, Y, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    cm = confusion_matrix(Y, lr.predict(X))
    plt.figure(figsize=(10, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig("conf_matrix.png")


def ro_curve(X, Y):
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(Y, lr.predict_proba(X)[:,1], pos_label=1)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')

if __name__ == '__main__':
    data = pd.read_csv('credit.clean', sep='\t', index_col=None)
    # CALC
    X = data.loc[:, data.columns.difference(['TARGET'])].values
    y = data['TARGET'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    font = {'size' : 15}

    plt.rc('font', **font)

    plot_confusion_matrix(X, y, classes=['Non-churned', 'Churned'],
                        title='Confusion matrix')
    ro_curve(X, y)

    print (lr.predict_proba(X))
    print (len(lr.predict_proba(X)))
    print (lr.predict_proba(X)[1])

    result = pd.DataFrame({'target': y, 'predict': lr.predict(X)})
    print (result.sort_values(by=['target', 'predict'], ascending=False))
    # plt.show()

    err_train = np.mean(y_train != lr.predict(X_train))
    err_test  = np.mean(y_test  != lr.predict(X_test))
    print (err_train, err_test)