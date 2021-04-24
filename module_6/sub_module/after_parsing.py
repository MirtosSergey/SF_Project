'''
Функции для обработки данных после парсинга.
'''

import numpy as np

def extract_char_engine(x):
    '''
    Функция для обработки признака "char_engine".
    Вход:
    * Строка.
    Выход:
    * Объем, мощность и тип двигателя.
    '''
    #Если пропуск
    if str(x) == 'nan':
        return x
    #Если электродвигатель
    elif 'Электро' in x:
        x = x.replace('\\xa0','').replace(' / ',' ').replace('л.с.','').replace('кВт','')
        x = x.split(' ')
        p1 = x[0]
        p2 = x[1]
        p3 = x[2]
        return [p2, p1, p3]
    #Остальные типы двигателей
    else:
        x = x.split(' ')
        p1 = x[0]
        p2 = x[1]
        p3 = ''
        for i in x[2:]:
            p3 += i
        return [p1, p2, p3]


def extract_char_engine_test(x):
    '''
    Функция для обработки признака "char_engine" для тестовой выборки.
    Вход:
    * Строка.
    Выход:
    * Объем, мощность и тип двигателя.
    '''
    p3 = x[::-1].split(' ')[0][::-1]
    p2 = int(x[x.find('(')+1:x.find(')')-5])
    p1 = x[:x.find('(')]
    p1 = p1[p1.find('.')-1:p1.find('.')+2]
    #Если электродвигателль
    if p3 == 'электро':
        p2 *= 10
        p1 = np.nan
    #Остальные типы двигателей
    else:
        p1 = float(p1)
    return [p1,p2,p3]


def extract_equipment_test(x):
    '''
    Функция для обработки признака "equipment" и списка.
    Вход:
    * Строка.
    Выход:
    * Список equipment.
    '''
    #Если пропуск
    if str(x) == 'nan':
        return x
    #Остальное
    else:
        mas = []
        while True:
            equi = x[x.find('"')+1:x.find('":')]
            x = x[x.find(',')+1:]
            if len(mas) > 0:
                if equi != mas[-1]:
                    mas += [equi]
                else:
                    break
            else:
                mas += [equi]
        return mas

def extract_month(x):
    '''
    Функция для обработки признака "date" и извлечения месяца.
    Вход:
    * Строка.
    Выход:
    * Месяц.
    '''
    #Если пропуск
    if str(x) == 'nan':
        return np.nan
    #Остальное
    elif len(x) > 1:
        return x[1]
    
def extract_year(x):
    '''
    Функция для обработки признака "date" и извлечения года.
    Вход:
    * Строка.
    Выход:
    * Год.
    '''
    #Если пропуск
    if str(x) == 'nan':
        return np.nan
    #Остальное
    elif len(x) == 3:
        return x[2]
    #Если есть месяц и день, но нет года
    elif len(x) == 2:
        return '2021'
    
if __name__ == '__main__':
    pass