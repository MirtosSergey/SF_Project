'''
Функции для ML.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import copy as cp

def get_metrics(Y,Yp,text=''):
    '''
    Получение метрик.
    Вход:
    * Y - массив правдивых решений;
    * Yp - массив предсказанных решений;
    * text - дополнительное пояснение.
    Выход:
    * None.
    '''
    print(text)
    print('MAE:     {:.3f}'.format(mean_absolute_error(Y,Yp)))
    print('MSE:     {:.3f}'.format(mean_squared_error(Y,Yp)))
    print('RMSE:    {:.3f}'.format(mean_squared_error(Y,Yp)**0.5))
    print('MAPE:    {:.3f} %'.format(mean_absolute_percentage_error(Y,Yp)*100))
    pass

def show_importances_feature(model,cols):
    '''
    Вывод самых значимых признаков по результатам ML.
    Вход:
    * model - обучившиеся модель;
    * cols - название столбцов.
    Выход:
    * None.
    '''
    #Размер графика
    plt.rcParams['figure.figsize'] = (15,20)
    #Формирование графика
    cols.remove('price')
    cols.remove('Kaggle')
    #Наиболее важные признаки
    feat_importances = pd.Series(model.coef_, index=cols).nlargest(100)
    feat_importances.plot(kind='barh')
    plt.show()
    pass

def transform_data_nom(df,col_cat,col_bin,col_drop):
    '''
    Функция для преобразования номинативных переменных.
    Вход:
    * df - DataFrame;
    * col_cat - список столбцов с категориальными переменными;
    * col_cat - список столбцов с бинарными переменными;
    * col_drop - столбцы для удаления (которых нет в col_cat).
    Выход:
    * Преобразованный DataFrame.
    '''
    #Dummy-кодирование
    for col in col_cat:
        df = pd.concat([df, pd.get_dummies(df[col],prefix=col)], axis=1)
    #LabelEncoder
    for col in col_bin:
        df[col] = LabelEncoder().fit_transform(df[col])
    #Удаление
    df = df.drop(columns=col_cat+col_drop+['url']).copy()
    return df

def get_split(df,target,drop):
    '''
    Функция для разделения выборки на тренировочную и тестовую и отделению признаков.
    Вход:
    * df - DataFrame;
    * target - название целевого признака;
    * drop - название столбцов, которые надо дополнительно удалить.
    Выход:
    * Тренировочная, тестовая выборки.     
    '''
    df_train = df.query('Kaggle==0')
    df_test = df.query('Kaggle==1')
    X_train = df_train.drop(columns=['Kaggle',target,drop])
    Y_train = df_train[target]
    X_test = df_test.drop(columns=['Kaggle',target,drop])
    return X_train,Y_train,X_test

def get_scaler(X_train,X_test):
    '''
    Функция для стандартной нормализации выборок.
    Вход:
    * X_train - обучающая выборка;
    * X_test - тестовая выборка.
    Выход:
    * Нормализованные тренировочная и тестовая выборки.
    '''
    #Разделение признаков на числовой и номинативный
    col_num = X_train.select_dtypes(exclude=['uint8']).columns
    col_nom = X_train.select_dtypes(include=['uint8']).columns
    #Стандартизация
    Scaler = StandardScaler()
    Scaler.fit(X_train[col_num])
    X_train_num = Scaler.transform(X_train[col_num])
    X_test_num = Scaler.transform(X_test[col_num])
    X_train_nom = X_train[col_nom].values
    X_test_nom = X_test[col_nom].values
    X_train = np.hstack([X_train_num,X_train_nom])
    X_test = np.hstack([X_test_num,X_test_nom])
    return X_train, X_test

def train_val_test_split(df,target,drop,random_state,test_size=0.3):
    '''
    Функция для разделения выборок и стандартизации признаков.
    Вход:
    * df - DataFrame;
    * target - название целевого признака;
    * drop - название столбцов, которые надо дополнительно удалить;
    * random_state - random_state;
    * test_size - отношение размеров тестовой выборки и исходной.
    Выход:
    * Тренировочные, валидационные, тестовые выборки. 
    '''
    #Разделение выборок
    X_train,Y_train,X_test = get_split(df,target,drop)
    #Нормализация
    X_train, X_test = get_scaler(X_train, X_test)
    #Разделение на обучающую и валидационную
    if test_size != 1:
        X_trn,X_val,Y_trn,Y_val = train_test_split(X_train,Y_train,random_state=random_state,test_size=0.3,shuffle=True)
    else:
        X_trn,X_val,Y_trn,Y_val = X_train,X_train,Y_train,Y_train
    return X_trn,X_val,Y_trn,Y_val,X_test

def get_MLRegr(XYs,model_init,ylim=(9.8,17.5)):
    '''
    Функция для обучения N однотипных моделей на разных выборках.
    Вход:
    * XYs - список выборок;
    * model_init - обучаемая модель;
    * ylim - ОДЗ целевого признака.
    Выход:
    * Таблица с результатами;
    * обученные модели.
    '''
    res,models = [],[]
    for i, xy in enumerate(XYs):
        i += 1
        print('Построение модели для выборки {}/{}.'.format(i,len(XYs)))
        X_trn,X_val,Y_trn,Y_val = xy
        model = cp.copy(model_init)
        #Обучение модели
        model.fit(X_trn,Y_trn)
        #Предсказание
        Y_pred_trn = model.predict(X_trn)
        Y_pred_val = model.predict(X_val)
        #Пост-обработка - удаление отрицательных значений
        Y_pred_trn = np.array([x if x > ylim[0] else ylim[0] for x in Y_pred_trn])
        Y_pred_val = np.array([x if x > ylim[0] else ylim[0] for x in Y_pred_val])
        Y_pred_trn = np.array([x if x < ylim[1] else ylim[1] for x in Y_pred_trn])
        Y_pred_val = np.array([x if x < ylim[1] else ylim[1] for x in Y_pred_val])
        #Пост-обработка - удаление логорифмирования целевого признака
        Y_trn = np.exp(Y_trn)
        Y_val = np.exp(Y_val)
        Y_pred_trn = np.exp(Y_pred_trn)
        Y_pred_val = np.exp(Y_pred_val)
        #MAPE
        mape_trn = mean_absolute_percentage_error(Y_trn,Y_pred_trn)*100
        mape_val = mean_absolute_percentage_error(Y_val,Y_pred_val)*100
        #Анализ ошибок
        df_sub_trn = pd.DataFrame({'y':Y_trn,'y_':Y_pred_trn, '%': np.abs(Y_pred_trn-Y_trn)/Y_trn*100})
        df_sub_val = pd.DataFrame({'y':Y_val,'y_':Y_pred_val, '%': np.abs(Y_pred_val-Y_val)/Y_val*100})
        fig, axes = plt.subplots(2,2,figsize = (14,14), sharex=True)
        #Зависимость действительных значений и предсказанных
        sns.scatterplot(x='y',y='y_',data=df_sub_trn,ax=axes[0][0])
        sns.scatterplot(x='y',y='y_',data=df_sub_val,ax=axes[0][1])
        #Распределение ошибок
        sns.scatterplot(x='y',y='%',data=df_sub_trn,ax=axes[1][0])
        sns.scatterplot(x='y',y='%',data=df_sub_val,ax=axes[1][1])
        #Пределы графиков
        axes[0][0].set(ylim=(0, 1e7),xlim=(0, 1e7))
        axes[0][1].set(ylim=(0, 1e7),xlim=(0, 1e7))
        axes[1][0].set(ylim=(0, 100),xlim=(0, 1e7))
        axes[1][1].set(ylim=(0, 100),xlim=(0, 1e7))
        #Название графика
        fig.suptitle('Распределение ошибок для выборки: {}'.format(i))
        axes[0][0].set_title('Y_(Y) trn')
        axes[0][1].set_title('Y_(Y) val')
        axes[1][0].set_title('% (Y) trn')
        axes[1][1].set_title('% (Y) val')
        #Общие результаты
        res += [[i,mape_trn,mape_val]]
        models += [model]
    tabl_res = pd.DataFrame(res,columns=['i','mape_trn','mape_val'])
    print(tabl_res)
    return tabl_res,models

if __name__ == '__main__':
    pass