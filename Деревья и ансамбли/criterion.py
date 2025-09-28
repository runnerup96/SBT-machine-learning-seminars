import numpy as np
from utils import sampling_proba


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array типа float размера (n_objects, n_classes)
            one-hot-encoding представление меток классов выборки
    
    Параметры
    -------
    float
        Энтропия входного набора данных
    """
    if len(y) == 0:
        return np.inf

    eps = 1e-4
    p_i = sampling_proba(y)
    return -np.sum(p_i * np.log2(p_i + eps))



def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Параметры
    ----------
    y : np.array типа float размера (n_objects, n_classes)
            one-hot-encoding представление меток классов выборки
    
    Возвращает
    -------
    float
        Индекс Джини входного набора данных
    """
    if len(y) == 0:
        return np.inf

    p_i = sampling_proba(y)
    return 1 - np.sum(p_i**2)



def variance(y):
    """
    Вычисляет дисперсию целевых значений переданной выборки 
    
    Параметры
    ----------
    y : np.array типа float с формой (n_objects, 1)
        Вектор целевых значений
    
    Возвращает
    -------
    float
        Дисперсия переданного вектора целевых значений
    """

    if len(y) == 0:
        return np.inf
    
    m = np.mean(y)
    return np.mean((y - m)**2)



def mad_median(y):
    """
    Вычисляет среднее абсолютное отклонение от медианы целевых значений переданной выборки
    
    Параметры
    ----------
    y : np.array типа float с формой (n_objects, 1)
        Вектор целевых значений
    
    Возвращает
    -------
    float
        Среднее абсолютное отклонение от медианы вектора целевых значений 
    """

    if len(y) == 0:
        return np.inf

    med = np.median(y)
    return np.mean(abs(y - med))