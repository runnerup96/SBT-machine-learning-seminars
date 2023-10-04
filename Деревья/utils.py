import numpy as np


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
