import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectPercentile


def constant_feature_detect(data, threshold=0.98):
    """
    Обнаружение признаков, которые показывают одинаковое значение для большинства/всех наблюдений (постоянные/почти постоянные признаки).

    Параметры
    ----------
    data : pd.DataFrame
        Входные данные в формате Pandas DataFrame.
    threshold : float, optional
        Порог для определения переменной как постоянной. Значение по умолчанию: 0.98.

    Возвращает
    -------
    list
        Список названий переменных, которые считаются почти постоянными.
    """

    # Создание копии данных для избежания изменения исходных данных
    data_copy = data.copy(deep=True)

    # Список для хранения имен почти постоянных переменных
    quasi_constant_feature = []

    # Итерация по всем признакам в данных
    for feature in data_copy.columns:
        # Вычисление доминирующего значения и его частоты
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]

        # Если доминирующее значение больше или равно заданному порогу, добавляем признак в список почти постоянных
        if predominant >= threshold:
            quasi_constant_feature.append(feature)

    # Вывод количества обнаруженных почти постоянных переменных
    print(len(quasi_constant_feature), ' переменных считаются почти постоянными')
    
    return quasi_constant_feature



def corr_feature_detect(data, threshold=0.8):
    """
    Обнаружение высококоррелированных признаков в DataFrame.

    Параметры
    ----------
    data : pd.DataFrame
        Входные данные в формате Pandas DataFrame.
    threshold : float, optional
        Порог для идентификации коррелированных переменных. Значение по умолчанию: 0.8.

    Возвращает
    -------
    list of pd.DataFrame
        Список групп коррелированных переменных в формате DataFrame.
    """

    # Вычисление матрицы корреляции между признаками
    corrmat = data.corr()

    # Взятие абсолютных значений коэффициентов корреляции
    corrmat = corrmat.abs().unstack()

    # Сортировка значений корреляции по убыванию
    corrmat = corrmat.sort_values(ascending=False)

    # Удаление значений корреляции равных 1 (диагональ)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1]

    # Преобразование результатов в DataFrame
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']

    # Инициализация списков для хранения групп коррелированных признаков
    grouped_feature_ls = []
    correlated_groups = []

    # Поиск коррелированных групп признаков
    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:

            # Находим все признаки, коррелированные с данным признаком
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(
                correlated_block.feature2.unique()) + [feature]

            # Добавляем блок коррелированных признаков в список
            correlated_groups.append(correlated_block)

    return correlated_groups


    
        