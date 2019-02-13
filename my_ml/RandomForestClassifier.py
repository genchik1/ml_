import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('credit.clean', sep='\t', index_col=None)
# CALC
X = data.loc[:, data.columns.difference(['TARGET'])].values
Y = data['TARGET'].values

cv_scores=[]
for k in range(1,101, 1):
    print (k)
    clf = RandomForestClassifier(n_estimators=k, random_state=56, verbose=0, n_jobs=30)
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, X, Y, cv=kf, scoring='roc_auc')
    cv_scores.append([k, scores.mean(), scores.std()])
df_scores = pd.DataFrame(cv_scores, columns=['k', 'roc_auc_score', 'STD'])

max_ = df_scores[df_scores['roc_auc_score']==df_scores['roc_auc_score'].max()]['k'].max()
print ('max: ', max_)

rf = RandomForestClassifier(n_estimators=max_, random_state=42, verbose=0, n_jobs=30)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
parameters = {
    'n_estimators': np.arange(50,400, 1),
    'max_depth': [1,3,5,7, 9, 11],
    'min_samples_split':[2,4,6, 8, 10],
    'min_samples_leaf':[2,4,6, 8, 10] }
clf = GridSearchCV(rf, parameters, cv=kf, scoring='roc_auc')
clf.fit(X, Y)

print (clf.best_params_ )

df = pd.DataFrame(clf.cv_results_)
df.to_csv('rf')