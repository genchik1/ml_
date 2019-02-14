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
    print (data[col].sort_values())
    if unique:
        print (data[col].unique())
    data[col].hist(bins=40)
    plt.plot()
    plt.show()


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
         

def personal_income(x):
    N = x['FACT_ADDRESS_PROVINCE'] + x['GEN_TITLE']
    if x['PERSONAL_INCOME']>=1000 and x['PERSONAL_INCOME']<300000 and x['GEN_INDUSTRY']==1:
        if x['PERSONAL_INCOME']>200000 and  N < .6:
            return np.nan
        elif x['PERSONAL_INCOME']>100000 and x['PERSONAL_INCOME']<=200000 and  N < .4:
            return np.nan
        elif x['PERSONAL_INCOME']>50000 and x['PERSONAL_INCOME']<=100000 and  N < .3:
            return np.nan
        elif x['PERSONAL_INCOME']>=1000 and x['PERSONAL_INCOME']<=50000 and  N < .2:
            return np.nan
        else:
            return x['PERSONAL_INCOME']
    else:
        return np.nan


def gen_industry(x):
    if x == 'Пропуск' or x == 'Другие сферы' or x==0:
        return 0
    else:
        return 1


def gen_title(data, col):
    m = {'Индивидуальный предприниматель':.6,
        'Партнер':.6,
        'Руководитель высшего звена':.5,
        'Руководитель среднего звена':.5,
        'Руководитель низшего звена':.4,
        'Военнослужащий по контракту':.3,
        'Высококвалифиц. специалист':.3,
        'Специалист':.2,
        'Служащий':.1,
        'Рабочий':.1,
        'Работник сферы услуг':.1,
        'Пропуск':np.nan,
        'Другое': np.nan,}
    data[col] = data[col].map(m)
    return data


def reg(x, m, col, ver=True):
    for l in m:
        if x[col] == l:
            return m[l]
    if ver:
        return 0
    else:
        return x[col]


def clean(data):
    
    data = gen_title(data, 'GEN_TITLE')
    data['GEN_TITLE'] = data['GEN_TITLE'].fillna(0)
    data['GEN_INDUSTRY'] = data.apply(lambda x: gen_industry(x['GEN_INDUSTRY']), axis=1)

    data['ORG_TP_FCAPITAL'] = data.apply(lambda x: 1 if x['ORG_TP_FCAPITAL'] == 'С участием' else 0, axis=1)

    data['JOB_DIR'] = data['JOB_DIR'].fillna(0)
    data['JOB_DIR'] = data.apply(lambda x: gen_industry(x['JOB_DIR']), axis=1)

    data['ORG_TP_STATE'] = data['ORG_TP_STATE'].fillna(0)
    data['ORG_TP_STATE'] = data.apply(lambda x: gen_industry(x['ORG_TP_STATE']), axis=1)

    m = {
        'Состою в браке':4,
        'Гражданский брак':3,
        'Разведен(а)':2,
        'Не состоял в браке':1,
        'Вдовец/Вдова':0
    }
    data["MARITAL_STATUS"] = data["MARITAL_STATUS"].map(m)

    m = {'Неполное среднее':0, 'Среднее':.1, 'Среднее специальное':.2,
        'Неоконченное высшее':.3, 'Высшее':.4, 'Два и более высших образования':.5,
        'Ученая степень':.6}

    data["EDUCATION"] = data["EDUCATION"].map(m)
    data["EDUCATION"] = data["EDUCATION"].fillna(0)


    m = {
        'Агинский Бурятский АО':'Бурятия',
        'Башкирия':'Башкортостан',
        'Коми-Пермяцкий АО':'Коми',
        'Марийская республика':'Марий Эл',
        'Мордовская республика':'Мордовия',
        'Пермская область':'Пермский край',
        'Санкт-Петербург':'Ленинградская область',
        'Усть-Ордынский Бурятский АО':'Бурятия',
        'Читинская область':'Бурятия',
        'Эвенкийский АО':'Красноярский край',
    }
    for col in ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE']:
        data[col] = data.apply(lambda x: reg(x, m, col, False), axis=1)

    # print (data[['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE']])
    m = {
        'Алтайский край':.1,
        'Башкортостан':.1,
        'Вологодская область':.1,
        'Иркутская область':.1,
        'Кемеровская область':.1,
        'Краснодарский край':.1,
        'Красноярский край':.1,
        'Ленинградская область':.1,
        'Москва':.1,
        'Московская область':.1,
        'Нижегородская область':.1,
        'Новосибирская область':.1,
        'Омская область':.1,
        'Оренбургская область':.1,
        'Пермский край':.1,
        'Ростовская область':.1,
        'Самарская область':.1,
        'Татарстан':.1,
        'Челябинская область':.1,
    }
    for col in ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE']:
        data[col] = data.apply(lambda x: reg(x, m, col), axis=1)

    m = {
        'ЦЕНТРАЛЬНЫЙ 1':.1,
        'ЦЕНТРАЛЬНЫЙ 2':.1,
        'УРАЛЬСКИЙ':.1,
        'ЦЕНТРАЛЬНЫЙ ОФИС':.1
    }
    data['REGION_NM'] = data.apply(lambda x: reg(x, m, 'REGION_NM'), axis=1)


    data['FACT_LIVING_TERM'] = data.apply(lambda x: fact_living_term(x), axis=1)
    data['FACT_LIVING_TERM'] = data['FACT_LIVING_TERM'].fillna(data['FACT_LIVING_TERM'].mean())

    data['WORK_TIME'] = data.apply(lambda x: clean_work_time(x), axis=1)
    data['WORK_TIME'] = data['WORK_TIME'].fillna(data['WORK_TIME'].mean())


    data['PERSONAL_INCOME'] = data.apply(lambda x: x['PERSONAL_INCOME'] if x['PERSONAL_INCOME']>1000 and x['PERSONAL_INCOME']<300000 else np.nan, axis=1)
    data['PERSONAL_INCOME'] = data.apply(lambda x: personal_income(x), axis=1)

    # valid(data, 'PERSONAL_INCOME', unique=True)
    data['PERSONAL_INCOME'] = data.apply(lambda x: data[data['AGE']==x['AGE']].groupby(['AGE'])['PERSONAL_INCOME'].mean() if x['PERSONAL_INCOME'] is None else x['PERSONAL_INCOME'], axis=1)


    data['PERSONAL_INCOME'] = data['PERSONAL_INCOME'].fillna(data['PERSONAL_INCOME'].median())

    # valid(data, 'PERSONAL_INCOME', unique=True)

    
    data['FAMILY_INCOME'] = data.apply(lambda x: family_income(x), axis=1)
    un = sorted(data['FAMILY_INCOME'].unique())
    m = dict([(u, i) for i, u in enumerate(un, 0)])
    data['FAMILY_INCOME'] = data['FAMILY_INCOME'].map(m)


    data["PREVIOUS_CARD_NUM_UTILIZED"] = data["PREVIOUS_CARD_NUM_UTILIZED"].fillna(0)
    return data


if __name__ == '__main__':
    name1 = 'Credit_new.csv'
    name = 'Credit.csv'
    name_out1 = 'Credit_new.clean'
    name_out = 'Credit_test.clean'


    data = pd.read_csv(name1, sep=';', encoding='CP1251', decimal=',')
    # data = data.drop(["TARGET"], axis=1)
    # df = pd.read_csv(name1, sep=';', encoding='CP1251', decimal=',')
    # data = pd.concat([data, df])

    # valid(data, 'DL_DOCUMENT_FL', unique=True)
    
    # del data['AGREEMENT_RK']

    data = clean(data)

    data.to_csv(name_out1, sep='\t', index=None)
    print (data.head())
    print ('save')
