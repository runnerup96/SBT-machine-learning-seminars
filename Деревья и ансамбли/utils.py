import numpy as np
from sklearn.preprocessing import LabelEncoder


def sampling_proba(y):
    """
    Вычисляет выборочные вероятности каждого классов

    Параметры
    ----------
    y : np.array типа float размера (n_objects, n_classes)
        one-hot-encoding представление меток классов выборки

    """
    n_i = np.sum(y, axis=0)
    n = np.sum(n_i)
    p_i = n_i / n
    return p_i


def feature_encoding (df, cat_cols):
    """
    Перекодирует категориальные фичи в числовой формат, нумеруя категории

    Параметры
    ----------
    
    df : pd.DataFrame
        Выборка
    cat_cols : list
        Список названий категориальных фичей
        
    Возвращает
    ----------
    
    df : pd.DataFrame
        Выборка с перекодированными фичами cat_cols
    label_encoder : dict
        Словарь кодировщиков категорий в числа
    """
    
    label_encoders = {}
    for column in cat_cols:
        le=LabelEncoder()
        df[column]=le.fit_transform(df[column])
        label_encoders[column]=le
    return df, label_encoders


def one_hot_encode(n_classes, y):
    return np.eye(n_classes, dtype=np.float64)[y]

def one_hot_decode(y_ohe):
    return y_ohe.argmax(axis=1, keepdims=True)


class Node:
    """
    Класс узла дерева решений
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        
        
def sigmoida(x):
    return 1. / (1 + np.exp(- x))

def softmax(x):
    e = np.exp(x) 
    return e / e.sum(axis=0)