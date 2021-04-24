'''
Функции для анализа данных.
'''

import matplotlib.pyplot as plt
import seaborn as sns

def show_boxplot(x,y,df,hue=None,size=(10,10)):
    '''
    Вывод графиков Box-Plot для параметра Y по группам X.
    Вход:
    * x - название столбца в DataFrame, по которому группируются данные;
    * y - название столбца в DataFrame, по которому строятся Box-Plot;
    * df - DataFrame;
    * hue - название столбца в DataFrame, по которому дополнительно разбиваются данные;
    * size - размер графика.
    Выход:
    * None.
    '''
    #Место для Line-Plot и его настройки
    fig, ax = plt.subplots(figsize = size)
    #Line-Plot
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    #Разворот подписи по оси X
    plt.xticks(rotation=90)
    #Описание над графиком
    ax.set_title('BoxPlot for ' + x)
    #Обновление графика
    plt.show()
    pass

def show_histplot(x,df,hue=None,size=(8,4),kde=False, bins=10):
    '''
    Вывод графиков Hist-Plot по абциссе X по группам hue.
    Вход:
    * x - название столбца в DataFrame, являющегося абсциссой графика;
    * df - DataFrame;
    * hue - название столбца в DataFrame, по которому дополнительно разбиваются данные;
    * size - размер графика;
    * kde - проводить автоматическую регрессию;
    * bins - количество разделений по гистограмме.
    Выход:
    * None.
    '''
    #Место для Hist-Plot и его настройки
    fig, ax = plt.subplots(figsize = size)
    #Hist-Plot
    sns.histplot(x=x, data=df, ax=ax, hue=hue, element='bars', kde=kde, bins=bins)
    #Разворот подписи по оси X
    plt.xticks(rotation=90)
    #Описание над графиком
    ax.set_title('Hist-Plot for ' + x)
    #Обновление графика
    plt.show()
    pass

def show_lineplot(x,y,df,hue=None,markers=True,size=(8,8)):
    '''
    Вывод графиков Line-Plot для Y(X) по группам hue.
    Вход:
    * x - название столбца в DataFrame, абциссы;
    * y - название столбца в DataFrame, ординаты;
    * df - DataFrame;
    * markers - показывать ли маркеры;
    * hue - название столбца в DataFrame, по которому дополнительно разбиваются данные;
    * size - размер графика.
    Выход:
    * None.
    '''
    #Место для Line-Plot и его настройки
    fig, ax = plt.subplots(figsize = size)
    #Line-Plot
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=hue, markers=True, ax=ax)
    #Разворот подписи по оси X
    plt.xticks(rotation=90)
    #Описание над графиком
    ax.set_title('LinePlot for ' + y)
    #Обновление графика
    plt.show()
    pass

def show_scatterplot(x,y,df,hue=None,size=(8,8)):
    '''
    Вывод графиков Scatter-Plot для Y(X) по группам hue.
    Вход:
    * x - название столбца в DataFrame, абциссы;
    * y - название столбца в DataFrame, ординаты;
    * df - DataFrame;
    * hue - название столбца в DataFrame, по которому дополнительно разбиваются данные;
    * size - размер графика.
    Выход:
    * None.
    '''
    #Место для Scatter-Plot и его настройки
    fig, ax = plt.subplots(figsize = size)
    #Scatter-Plot
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=hue, ax=ax)
    #Разворот подписи по оси X
    plt.xticks(rotation=90)
    #Описание над графиком
    ax.set_title('ScatterPlot for ' + y)
    #Обновление графика
    plt.show()
    pass

def show_Heatmap(df,cols=None,size=(20,20)):
    '''
    Построение тепловой карты по матрице корреляций.
    Вход:
    * df - DataFrame;
    * cols - столбцы в DataFrame, по которым считаются корреляции;
    * size - размер графика.
    Выход:
    * None.
    '''
    if cols==None:
        cols=df.columns
    #Таблица корреляций
    table_corr = df[cols].corr().round(2)
    #Место для Line-Plot и его настройки
    fig, ax = plt.subplots(figsize = size)
    #Heat-Map
    sns.heatmap(table_corr, vmin=-1, vmax=1, cmap="YlGnBu",annot=True)
    #Описание над графиком
    ax.set_title('HeatMap')
    #Обновление графика
    plt.show()
    pass

if __name__ == '__main__':
    pass