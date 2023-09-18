import pandas as pd
# import numpy as np
# from warnings import warn

class GroupingRareValues():
    """
    Группировка наблюдений, содержащих редкие метки, в уникальную категорию ('rare').

    Parameters
    ----------
    mapping : list, optional
        Список сопоставлений меток столбцов. Каждое сопоставление представлено словарем с ключами 'col' (имя столбца)
        и 'mapping' (словарь сопоставления значений). По умолчанию None.
    cols : list, optional
        Список столбцов, для которых будет выполнена группировка редких меток. По умолчанию None.
    threshold : float, optional
        Порог для определения редких меток. Метки, встречающиеся реже, чем threshold, считаются редкими.
        По умолчанию 0.01.

    Attributes
    ----------
    cols : list
        Список столбцов, для которых будет выполнена группировка редких меток.
    mapping : list
        Список сопоставлений меток столбцов.

    Methods
    -------
    fit(X, y=None, **kwargs)
        Обучение кодировщика на основе данных X и y.
    transform(X)
        Преобразование новых категориальных данных с применением кодировки.
    grouping(X_in, threshold, mapping=None, cols=None)
        Группировка наблюдений, содержащих редкие метки, в уникальную категорию ('rare').

    """

    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        """
        Обучение кодировщика на основе данных X и y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Матрица признаков, где n_samples - количество образцов,
            а n_features - количество признаков.
        y : array-like, shape = [n_samples]
            Целевые значения.

        Returns
        -------
        self : encoder
            Возвращает self.

        """
        self._dim = X.shape[1]

        _, categories = self.grouping(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )
        self.mapping = categories
        return self

    def transform(self, X):
        """
        Выполнение преобразования новых категориальных данных.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Матрица новых данных, где n_samples - количество образцов,
            а n_features - количество признаков.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features]
            Преобразованные значения с примененной кодировкой.

        """
        if self._dim is None:
            raise ValueError('Необходимо обучить кодировщик перед его использованием для преобразования данных.')

        # Убедимся, что размерность входных данных соответствует ожидаемой.
        if X.shape[1] != self._dim:
            raise ValueError('Неожиданная размерность входных данных %d, ожидалась %d' % (X.shape[1], self._dim,))

        X, _ = self.grouping(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )

        return X

    def grouping(self, X_in, threshold, mapping=None, cols=None):
        """
        Группировка наблюдений, содержащих редкие метки, в уникальную категорию ('rare').

        Parameters
        ----------
        X_in : DataFrame
            Входные данные для группировки.
        threshold : float
            Порог для определения редких меток. Метки, встречающиеся реже, чем threshold, считаются редкими.
        mapping : list, optional
            Список сопоставлений меток столбцов. По умолчанию None.
        cols : list, optional
            Список столбцов, для которых будет выполнена группировка редких меток. По умолчанию None.

        Returns
        -------
        X : DataFrame
            Копия входных данных с примененной группировкой.
        mapping_out : list
            Список сопоставлений меток столбцов после группировки.

        """
        X = X_in.copy(deep=True)

        if mapping is not None:  # Применение существующих сопоставлений
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')  # Получение имени столбца
                X[column] = X[column].map(i['mapping'])
        else:  # Обучение новых сопоставлений
            mapping_out = []
            for col in cols:
                temp_df = pd.Series(X[col].value_counts() / len(X))
                mapping = {k: ('rare' if k not in temp_df[temp_df >= threshold].index else k)
                          for k in temp_df.index}
                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out


class ModeImputation():
    """
    Замена редких меток на наиболее часто встречающуюся метку.

    Parameters
    ----------
    mapping : list, optional
        Список сопоставлений меток столбцов. Каждое сопоставление представлено словарем с ключами 'col' (имя столбца)
        и 'mapping' (словарь сопоставления значений). По умолчанию None.
    cols : list, optional
        Список столбцов, для которых будет выполнена замена редких меток. По умолчанию None.
    threshold : float, optional
        Порог для определения редких меток. Метки, встречающиеся реже, чем threshold, считаются редкими.
        По умолчанию 0.01.

    Attributes
    ----------
    cols : list
        Список столбцов, для которых будет выполнена замена редких меток.
    mapping : list
        Список сопоставлений меток столбцов.

    Methods
    -------
    fit(X, y=None, **kwargs)
        Обучение кодировщика на основе данных X и y.
    transform(X)
        Преобразование новых категориальных данных с применением кодировки.
    impute_with_mode(X_in, threshold, mapping=None, cols=None)
        Замена редких меток на наиболее часто встречающуюся метку.

    """

    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        """
        Обучение кодировщика на основе данных X и y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Матрица признаков, где n_samples - количество образцов,
            а n_features - количество признаков.
        y : array-like, shape = [n_samples]
            Целевые значения.

        Returns
        -------
        self : encoder
            Возвращает self.

        """
        self._dim = X.shape[1]

        _, categories = self.impute_with_mode(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )
        self.mapping = categories
        return self

    def transform(self, X):
        """
        Выполнение преобразования новых категориальных данных.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Матрица новых данных, где n_samples - количество образцов,
            а n_features - количество признаков.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features]
            Преобразованные значения с примененной кодировкой.

        """
        if self._dim is None:
            raise ValueError('Необходимо обучить кодировщик перед его использованием для преобразования данных.')

        # Убедимся, что размерность входных данных соответствует ожидаемой.
        if X.shape[1] != self._dim:
            raise ValueError('Неожиданная размерность входных данных %d, ожидалась %d' % (X.shape[1], self._dim,))

        X, _ = self.impute_with_mode(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )

        return X

    def impute_with_mode(self, X_in, threshold, mapping=None, cols=None):
        """
        Замена редких меток на наиболее часто встречающуюся метку.

        Parameters
        ----------
        X_in : DataFrame
            Входные данные для замены редких меток.
        threshold : float
            Порог для определения редких меток. Метки, встречающиеся реже, чем threshold, считаются редкими.
        mapping : list, optional
            Список сопоставлений меток столбцов. Каждое сопоставление представлено словарем с ключами 'col' (имя столбца)
            и 'mapping' (словарь сопоставления значений). По умолчанию None.
        cols : list, optional
            Список столбцов, для которых будет выполнена замена редких меток. По умолчанию None.

        Returns
        -------
        X : DataFrame
            Исходные данные с выполненной заменой редких меток.
        mapping_out : list
            Список сопоставлений меток столбцов.

        """
        X = X_in.copy(deep=True)

        if mapping is not None:  # Применение сопоставлений, если они доступны
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')  # Получение имени столбца
                X[column] = X[column].map(i['mapping'])
        else:  # Обучение новых сопоставлений
            mapping_out = []
            for col in cols:
                temp_df = pd.Series(X[col].value_counts() / len(X))
                mode = X[col].mode()[0]
                mapping = {k: (mode if k not in temp_df[temp_df >= threshold].index else k)
                          for k in temp_df.index}
                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out

