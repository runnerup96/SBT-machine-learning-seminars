import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab


def diagnostic_plots(df, variable):
    """
    Функция для построения гистограммы и графика Q-Q (квантиль-квантиль)
    рядом для заданной переменной.

    Параметры:
    df (DataFrame): Исходный набор данных.
    variable (str): Название переменной, для которой требуется построить графики.

    Возвращаемое значение:
    Нет (функция отображает графики, но не возвращает значения).
    """
    
    # Создаем фигуру для графиков
    plt.figure(figsize=(15,6))
    
    # Первый подграфик - гистограмма
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=50)

    # Второй подграфик - график Q-Q
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    # Отображаем графики
    plt.show()

    
    
def log_transform(data, cols=[]):
    """
    Логарифмическое преобразование данных.

    Параметры:
    data (DataFrame): Исходный набор данных.
    cols (list): Список переменных, для которых требуется применить логарифмическое преобразование.

    Возвращаемое значение:
    data_copy (DataFrame): Копия исходного набора данных с добавленными переменными,
    подвергнутыми логарифмическому преобразованию.
    """
    
    # Создаем копию исходных данных
    data_copy = data.copy(deep=True)
    
    # Применяем логарифмическое преобразование к указанным переменным
    for i in cols:
        data_copy[i+'_log'] = np.log(data_copy[i]+1)
        
        # Выводим график Q-Q для преобразованной переменной
        print('График Q-Q для переменной ' + i + '_log')
        diagnostic_plots(data_copy, str(i+'_log'))
        
    return data_copy



def reciprocal_transform(data, cols=[]):
    """
    Обратное преобразование данных.

    Параметры:
    data (DataFrame): Исходный набор данных.
    cols (list): Список переменных, для которых требуется применить обратное преобразование.

    Возвращаемое значение:
    data_copy (DataFrame): Копия исходного набора данных с добавленными переменными,
    подвергнутыми обратному преобразованию.
    """
    
    # Создаем копию исходных данных
    data_copy = data.copy(deep=True)
    
    # Применяем обратное преобразование к указанным переменным
    for i in cols:
        data_copy[i+'_reciprocal'] = 1/(data_copy[i])
        
        # Выводим график Q-Q для преобразованной переменной
        print('График Q-Q для переменной ' + i + '_reciprocal')
        diagnostic_plots(data_copy, str(i+'_reciprocal'))
        
    return data_copy



def square_root_transform(data, cols=[]):
    """
    Квадратный корень (square root) преобразование данных.

    Параметры:
    data (DataFrame): Исходный набор данных.
    cols (list): Список переменных, для которых требуется применить квадратный корень.

    Возвращаемое значение:
    data_copy (DataFrame): Копия исходного набора данных с добавленными переменными,
    подвергнутыми квадратному корню.
    """
    
    # Создаем копию исходных данных
    data_copy = data.copy(deep=True)
    
    # Применяем квадратный корень к указанным переменным
    for i in cols:
        data_copy[i+'_square_root'] = (data_copy[i])**(0.5)
        
        # Выводим график Q-Q для преобразованной переменной
        print('График Q-Q для переменной ' + i + '_square_root')
        diagnostic_plots(data_copy, str(i+'_square_root'))        
    return data_copy 


def exp_transform(data, coef, cols=[]):
    """
    Экспоненциальное (exp) преобразование данных.

    Параметры:
    data (DataFrame): Исходный набор данных.
    coef (float): Коэффициент степени для преобразования.
    cols (list): Список переменных, для которых требуется применить экспоненциальное преобразование.

    Возвращаемое значение:
    data_copy (DataFrame): Копия исходного набора данных с добавленными переменными,
    подвергнутыми экспоненциальному преобразованию.
    """
    
    # Создаем копию исходных данных
    data_copy = data.copy(deep=True)
    
    # Применяем экспоненциальное преобразование с заданным коэффициентом к указанным переменным
    for i in cols:
        data_copy[i+'_exp'] = (data_copy[i])**coef
        
        # Выводим график Q-Q для преобразованной переменной
        print('График Q-Q для переменной ' + i + '_exp')
        diagnostic_plots(data_copy, str(i+'_exp'))         
    return data_copy 

