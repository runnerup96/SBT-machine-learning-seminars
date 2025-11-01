import pandas as pd

class MeanEncoding():
    """
    Класс для замены метки средним значением целевой переменной для данной метки.

    Параметры
    ----------
    mapping : list, по умолчанию None
        Список маппингов для преобразования меток. Каждый маппинг - это словарь,
        содержащий 'col' (имя столбца) и 'mapping' (словарь для замены значений).
    cols : list, по умолчанию None
        Список столбцов, для которых будет применяться кодирование.
    """

    def __init__(self, mapping=None, cols=None):
        self.cols = cols
        self.mapping = mapping
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        """
        Обучение кодировщика на основе данных X и y.

        Параметры
        ----------
        X : датафрейм, форма = [n_samples, n_features]
            Обучающие векторы, где n_samples - количество выборок,
            а n_features - количество признаков.
        y : список целевых значений, форма = [n_samples]
            Значения целевой переменной.

        Возвращает
        -------
        self : кодировщик
            Возвращает сам объект кодировщика.
        """
        self._dim = X.shape[1]

        _, categories = self.mean_encoding(
            X,
            y,
            mapping=self.mapping,
            cols=self.cols
        )
        self.mapping = categories
        return self

    def transform(self, X):
        """
        Применение преобразования к новым категориальным данным.

        Использует маппинг (если доступен) и список столбцов для кодирования данных.

        Параметры
        ----------
        X : датафрейм, форма = [n_samples, n_features]

        Возвращает
        -------
        X : Преобразованные значения с примененным кодированием.
        """
        if self._dim is None:
            raise ValueError('Необходимо обучить кодировщик перед его использованием для преобразования данных.')

        if X.shape[1] != self._dim:
            raise ValueError('Неожиданное значение размерности входных данных %d, ожидалось %d' % (X.shape[1], self._dim,))

        X, _ = self.mean_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols
        )

        return X

    def mean_encoding(self, X_in, y=None, mapping=None, cols=None):
        """
        Группировка наблюдений с редкими метками в уникальную категорию ('rare').

        Параметры
        ----------
        X_in : DataFrame
            Входные данные.
        y : массивоподобный, форма = [n_samples], по умолчанию None
            Значения целевой переменной.
        mapping : list, по умолчанию None
            Список маппингов для преобразования меток. Каждый маппинг - это словарь,
            содержащий 'col' (имя столбца) и 'mapping' (словарь для замены значений).
        cols : list, по умолчанию None
            Список столбцов для кодирования.

        Возвращает
        -------
        X : DataFrame
            Преобразованные данные.
        mapping_out : list
            Список маппингов для кодирования.
        """
        X = X_in.copy(deep=True)

        if mapping is not None:
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')  # Получаем имя столбца
                X[column] = X[column].map(i['mapping'])
        else:
            mapping_out = []
            for col in cols:
                mapping = X[y.name].groupby(X[col]).mean().to_dict()#группировка по категориальной переменной и рассчет среднего целевой
                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out
