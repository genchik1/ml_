import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# визуализация
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(12,8)})
sns.set(font_scale=1.5)
sns.set_color_codes("muted")

plt.figure(figsize=(10, 8))


def changing_values_to_boolean(data, col):
    data[col] = data.apply(lambda x: 1 if x[col] is not None else 0, axis=1)
    return data


def clean_work_time(x):
    if (x['AGE']-14)*12 >= x['WORK_TIME'] and x['WORK_TIME'] is not None:
        return x['WORK_TIME']
    else:
        return np.nan


def fact_living_term(x):
    if x['FACT_LIVING_TERM'] > (x['AGE']+1)*12:
        return np.nan
    else:
        return x['FACT_LIVING_TERM']


def valid(data, col, unique=False):
    print (data[col])
    if unique:
        print (data[col].unique())
    data[col].hist(bins=40)
    plt.plot()


def family_income(x):
    str_ = x['FAMILY_INCOME']
    if str_.split(' ')[0] == 'до':
        y = [0, int(str_.split(' ')[1])]
    elif str_.split(' ')[0] == 'свыше':
        y = [int(str_.split(' ')[1]), 1000000000]
    else:
        str_ = str_.split(' ')
        y = [int(str_[1]), int(str_[3])]
    
    if x['PERSONAL_INCOME'] <= y[-1]:
        return y[-1]
    else:
        if x['PERSONAL_INCOME'] < 10000:
            return 10000
        elif x['PERSONAL_INCOME'] < 20000:
            return 20000
        elif x['PERSONAL_INCOME'] < 50000:
            return 50000
        else:
            return 100000
         

if __name__ == '__main__':
    data = pd.read_csv('Credit.csv', sep=';', encoding='CP1251', decimal=',')

    print (data.head())
    # print (data.info())

    # Уберем ID пользователей
    data = data.drop("AGREEMENT_RK", axis=1)

    data['FACT_LIVING_TERM'] = data.apply(lambda x: fact_living_term(x), axis=1)
    data['FACT_LIVING_TERM'] = data['FACT_LIVING_TERM'].fillna(data['FACT_LIVING_TERM'].mean())

    data['WORK_TIME'] = data.apply(lambda x: clean_work_time(x), axis=1)
    data['WORK_TIME'] = data['WORK_TIME'].fillna(data['WORK_TIME'].mean())

    data['PERSONAL_INCOME'] = data.apply(lambda x: x['PERSONAL_INCOME'] if x['PERSONAL_INCOME']>1000 else np.nan, axis=1)
    data['PERSONAL_INCOME'] = data['PERSONAL_INCOME'].fillna(data['PERSONAL_INCOME'].median())

    m = {'Неполное среднее':0, 'Среднее':1, 'Среднее специальное':2,
        'Неоконченное высшее':3, 'Высшее':4, 'Два и более высших образования':5,
        'Ученая степень':6}

    data["EDUCATION"] = data["EDUCATION"].map(m)

    
    data['FAMILY_INCOME'] = data.apply(lambda x: family_income(x), axis=1)
    un = sorted(data['FAMILY_INCOME'].unique())
    m = dict([(u, i) for i, u in enumerate(un, 0)])
    data['FAMILY_INCOME'] = data['FAMILY_INCOME'].map(m)

    m = {
        'Состою в браке':4,
        'Гражданский брак':3,
        'Разведен(а)':2,
        'Не состоял в браке':1,
        'Вдовец/Вдова':0
    }
    data["MARITAL_STATUS"] = data["MARITAL_STATUS"].map(m)

    data["PREVIOUS_CARD_NUM_UTILIZED"] = data["PREVIOUS_CARD_NUM_UTILIZED"].fillna(0)

    data['ORG_TP_FCAPITAL'] = data.apply(lambda x: 1 if x['ORG_TP_FCAPITAL']=='С участием' else 0, axis=1)

    for cat in ['GEN_INDUSTRY', 'GEN_TITLE', 'ORG_TP_STATE', 'JOB_DIR', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE', 'REGION_NM']:
        data = changing_values_to_boolean(data, cat)


    # valid(data, 'FAMILY_INCOME', unique=True)

    # plt.show()

    # raise SystemExit

    data.to_csv('credit.clean', sep='\t', index=None)
