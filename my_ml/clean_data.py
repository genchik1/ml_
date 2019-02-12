import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def changing_values_to_boolean(data, col):
    data[col] = data.apply(lambda x: 1 if x[col] is not None else 0, axis=1)
    return data


def clean_work_time(x):
    if (x['AGE']-12)*12 >= x['WORK_TIME'] and x['WORK_TIME'] is not None:
        return True
    else:
        return False


def replace_work_time(mean_wt, x):
    if x['y'] == False:
        return mean_wt
    else:
        return x['WORK_TIME']


if __name__ == '__main__':
    data = pd.read_csv('Credit.csv', sep=';', encoding='CP1251')

    print (data.head())
    # print (data.info())

    # Уберем ID пользователей
    data = data.drop("AGREEMENT_RK", axis=1)


    m = {'Неполное среднее':0, 'Среднее':1, 'Среднее специальное':2,
        'Неоконченное высшее':3, 'Высшее':4, 'Два и более высших образования':5,
        'Ученая степень':6}

    data["EDUCATION"] = data["EDUCATION"].map(m)

    m = {
        'Состою в браке':4,
        'Гражданский брак':3,
        'Разведен(а)':2,
        'Не состоял в браке':1,
        'Вдовец/Вдова':0
    }
    data["MARITAL_STATUS"] = data["MARITAL_STATUS"].map(m)


    data = changing_values_to_boolean(data, "MARITAL_STATUS")

    data["PREVIOUS_CARD_NUM_UTILIZED"] = data["PREVIOUS_CARD_NUM_UTILIZED"].fillna(0)

    data['y'] = data.apply(lambda x: clean_work_time(x), axis=1)

    mean_wt = data[data['y']==True].WORK_TIME.mean()

    data['WORK_TIME'] = data.apply(lambda x: replace_work_time(mean_wt, x), axis=1)

    del data['y']

    data['ORG_TP_FCAPITAL'] = data.apply(lambda x: 1 if x['ORG_TP_FCAPITAL']=='С участием' else 0, axis=1)

    for cat in ['GEN_INDUSTRY', 'GEN_TITLE', 'ORG_TP_STATE', 'JOB_DIR', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE', 'REGION_NM']:
        data = changing_values_to_boolean(data, cat)

    for cat in ['PERSONAL_INCOME', 'CREDIT', 'FST_PAYMENT', 'LOAN_AVG_DLQ_AMT', 'LOAN_MAX_DLQ_AMT']:
        data[cat] = data[cat].str.replace(',', '.').astype(float)

    # object_column = [x for x in data.columns if data[x].dtypes == object]
    # print (object_column, '\n')
    # print (data[object_column].head())

    data['FAMILY_INCOME'] = data['FAMILY_INCOME'].str.split(' руб').str[0]
    data['FAMILY_INCOME'] = data['FAMILY_INCOME'].str.split(' ').str[-1]

    data['FAMILY_INCOME'] = data['FAMILY_INCOME'].astype(float)

    data['FAMILY_INCOME'] = data.apply(lambda x: x['PERSONAL_INCOME'] if x['FAMILY_INCOME']<x['PERSONAL_INCOME'] else x['FAMILY_INCOME'], axis=1)


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