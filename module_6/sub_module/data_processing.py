'''
Функции для постобработки данных.
'''

import sub_module.visualization as viz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ttest_ind

def intersection_data(df_train,df_test,cols):
    '''
    Функция для обнаружения отсутствующих значений в обучающей и тестовой выборках.
    Вход:
    * df_train - обучающая выборка;
    * df_test - тестовая выборка;
    * cols - столбцы в DataFrame.
    Выход:
    * None.
    '''
    for col in cols:
        #Уникальные значения
        unique_train = list(df_train[col].unique())
        unique_test = list(df_test[col].unique())
        #Поиск не совпадений
        dict_intersection = {'train':[],'test':[]}
        all_unique = unique_train + unique_test
        for un in all_unique:
            if un not in unique_train:
                dict_intersection['train'] += [un]
            elif un not in unique_test:
                dict_intersection['test'] += [un]
        #Вывод информации        
        print('Признак {}.'.format(col))
        print('    Пропуски в train: {}.'.format(dict_intersection['train']))
        print('    Пропуски в test:  {}.'.format(dict_intersection['test']))
        print('    Количество пропусков в train: {}'.format(df_train[col].isna().sum()))
        print('    Количество пропусков в test:  {}'.format(df_test[col].isna().sum()))
        print('    Количество уникальных элементов: {}'.format(len(set(all_unique))))
        print('    Уникальные элементы: {}'.format(set(all_unique)),end='\n'*2)
    pass

def count_drop_num(df,col):
    '''
    Функция для подсчета количества выбросов в DataFrame по числовому признаку col.
    Вход:
    * df - DataFrame;
    * col - столбец в DataFrame, по которому считаются выбросы.
    Выход:
    * Количество выбросов.
    '''
    #Квантили
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    #Межквантильный размах
    IQR = q75 - q25
    #Количество начальное
    len_init = len(df[col])
    #Количество конечно
    len_end = df[col].between(q25 - 1.5*IQR, q75 + 1.5*IQR).sum()
    return len_init - len_end

def Is_need_transform(df,col,func):
    '''
    Функция для вывода всей информации о необходимости преобразовании признака.
    Вход:
    * df - DataFrame;
    * col - столбец в DataFrame, по которому считается;
    * func - функция преобразования.
    Выход:
    * None.
    '''
    #Место для Hist-Plot и его настройки
    fig, axes = plt.subplots(1,2,figsize = (12,5))
    #Название
    axes[0].set_title('До преобразования')
    axes[1].set_title('После преобразования')
    #Графики
    #Изначальный
    df_sub = df.query('Kaggle==0').copy()
    drop_init = count_drop_num(df_sub,col)
    sns.histplot(x=col, data=df_sub, ax=axes[0], element='bars', bins=25)
    #Конечный
    df_sub[col] = df_sub[col].apply(func)
    drop_end = count_drop_num(df_sub,col)
    sns.histplot(x=col, data=df_sub, ax=axes[1], element='bars', bins=25)
    #Вывод информации
    print('Признак {} преобразован. Количество признаков до: {}; после {}; сокращено: {}.'.\
         format(col,drop_init,drop_end,drop_init-drop_end))
    #Описание над графиком
    fig.suptitle('Hist-Plot for ' + col + ' (train)')
    pass

def Is_drop(df,col):
    '''
    Функция для определения выброс прецендент или нет.
    Вход:
    * df - DataFrame;
    * col - столбец в DataFrame, по которому считаются выбросы.
    Выход:
    * Количество выбросов.
    '''
    #Квантили
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    #Межквантильный размах
    IQR = q75 - q25
    return ~df[col].between(q25 - 1.5*IQR, q75 + 1.5*IQR)

def Is_stat_dif(x,y,df,alpha=0.05):
    '''
    Поиск статически значимых параметров.
    Вход:
    * x - название столбца в DataFrame, по которому группируются данные;
    * y - название столбца в DataFrame, по которому считается доверительный интервал;
    * df - DataFrame;
    * alpha=0.05 - уровень значимости.
    Выход:
    * Список из элементов [статически не значим, статически значим].
    '''
    #Список групп
    ind = df.loc[:, x].value_counts().index
    #Создание различных комбинаций из списка по 2
    combo = list(combinations(ind, 2))
    #Поиск
    for comb in combo:
        #Определение p-уровня значимости
        p = ttest_ind(df.loc[df.loc[:, x] == comb[0], y],
                     df.loc[df.loc[:, x] == comb[1], y]).pvalue
        #Проверка (знаменатель необходим для учета поправки Бонферрони)
        if p <= alpha / len(combo):
            print('Статистически значим: {}'.format(x))
            return [np.nan, x]
    else:
        print('Статистически не значим: {}'.format(x))
        return [x, np.nan]
    pass

if __name__ == '__main__':
    pass