import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# from warnings import warn

# ChiMerge method modeified from https://github.com/tatsumiw/ChiMerge/blob/master/ChiMerge.py
# TODO: add more constraits to the discretized result.
class ChiMerge():
    """
    Дискретизация с учителем с использованием метода ChiMerge.
    
    Parameters
    ----------
    confidenceVal : number, optional
        Значение доверительного интервала для критерия хи-квадрат. По умолчанию 3.841, что соответствует p=0.05 и dof=1.
    num_of_bins : int, optional
        Количество интервалов после дискретизации. По умолчанию 10.
    col : str, optional
        Имя столбца, для которого будет выполнена дискретизация.

    Attributes
    ----------
    col : str
        Имя столбца, для которого будет выполнена дискретизация.
    confidenceVal : number
        Значение доверительного интервала для критерия хи-квадрат.
    bins : list
        Грани интервалов после дискретизации.
    num_of_bins : int
        Количество интервалов после дискретизации.

    Methods
    -------
    fit(X, y, **kwargs)
        Обучение дискретизатора на основе данных X и y.
    transform(X)
        Преобразование новых данных.
    chimerge(X_in, y=None, confidenceVal=None, num_of_bins=None, col=None, bins=None)
        Дискретизация переменной с использованием метода ChiMerge.

    """

    def __init__(self, col=None, bins=None, confidenceVal=3.841, num_of_bins=10):
        self.col = col
        self._dim = None
        self.confidenceVal = confidenceVal
        self.bins = bins
        self.num_of_bins = num_of_bins

    def fit(self, X, y, **kwargs):
        """
        Обучение дискретизатора на основе данных X и y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Обучающие векторы, где n_samples - количество образцов,
            а n_features - количество признаков.
        y : array-like, shape = [n_samples]
            Целевые значения.

        Returns
        -------
        self : encoder
            Возвращает self.

        """
        self._dim = X.shape[1]

        _, bins = self.chimerge(
            X_in=X,
            y=y,
            confidenceVal=self.confidenceVal,
            col=self.col,
            num_of_bins=self.num_of_bins
        )
        self.bins = bins
        return self

    def transform(self, X):
        """
        Преобразование новых данных.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X : новый dataframe с дискретизированным новым столбцом.

        """

        if self._dim is None:
            raise ValueError('Необходимо обучить дискретизатор перед его использованием для преобразования данных.')

        # Убедимся, что размерность входных данных соответствует ожидаемой.
        if X.shape[1] != self._dim:
            raise ValueError('Неожиданная размерность входных данных %d, ожидалась %d' % (X.shape[1], self._dim,))

        X, _ = self.chimerge(
            X_in=X,
            col=self.col,
            bins=self.bins
        )

        return X

    def chimerge(self, X_in, y=None, confidenceVal=None, num_of_bins=None, col=None, bins=None):
        """
        Дискретизация переменной с использованием метода ChiMerge.

        Parameters
        ----------
        X_in : DataFrame
            Входные данные для дискретизации.
        y : array-like, shape = [n_samples], optional
            Целевые значения. По умолчанию None.
        confidenceVal : number, optional
            Значение доверительного интервала для критерия хи-квадрат. По умолчанию None.
        num_of_bins : int, optional
            Количество интервалов после дискретизации. По умолчанию None.
        col : str, optional
            Имя столбца, для которого будет выполнена дискретизация. По умолчанию None.
        bins : list, optional
            Грани интервалов для дискретизации. По умолчанию None.

        Returns
        -------
        X : DataFrame
            Исходные данные с выполненной дискретизацией.
        bins : list
            Грани интервалов после дискретизации.

        """
        X = X_in.copy(deep=True)

        if bins is not None:  # Применение дискретизации, если грани интервалов уже определены
            try:
                X[col+'_chimerge'] = pd.cut(X[col], bins=bins, include_lowest=True)
            except Exception as e:
                print(e)
        else:  # Обучение новой дискретизации
            try:
                # Создание массива, который сохраняет количество образцов с 0/1 значением целевой переменной для столбца, который будет дискретизирован
                total_num = X.groupby([col])[y].count()
                total_num = pd.DataFrame({'total_num': total_num})
                positive_class = X.groupby([col])[y].sum()
                positive_class = pd.DataFrame({'positive_class': positive_class})
                regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True, how='inner')
                regroup.reset_index(inplace=True)
                regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']
                regroup = regroup.drop('total_num', axis=1)
                np_regroup = np.array(regroup)

                # Объединение интервалов, в которых 0 положительных или отрицательных образцов
                i = 0
                while (i <= np_regroup.shape[0] - 2):
                    if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                        np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # Положительные
                        np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # Отрицательные
                        np_regroup[i, 0] = np_regroup[i + 1, 0]
                        np_regroup = np.delete(np_regroup, i + 1, 0)
                        i = i - 1
                    i = i + 1

                # Вычисление значения хи-квадрат для соседних интервалов
                chi_table = np.array([])
                for i in np.arange(np_regroup.shape[0] - 1):
                    chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
                          * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
                          ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                                  np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
                    chi_table = np.append(chi_table, chi)

                # Объединение интервалов, у которых хи-квадрат близок
                while (1):
                    if (len(chi_table) <= (num_of_bins - 1) and min(chi_table) >= confidenceVal):
                        break
                    chi_min_index = np.argwhere(chi_table == min(chi_table))[0]
                    np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
                    np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
                    np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
                    np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

                    if (chi_min_index == np_regroup.shape[0] - 1):
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table = np.delete(chi_table, chi_min_index, axis=0)
                    else:
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                                   * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                                   ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                        chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)

                result_data = pd.DataFrame()
                result_data['variable'] = [col] * np_regroup.shape[0]
                bins = []
                tmp = []
                for i in np.arange(np_regroup.shape[0]):
                    if i == 0:
                        y = '-inf' + ',' + str(np_regroup[i, 0])
                    elif i == np_regroup.shape[0] - 1:
                        y = str(np_regroup[i - 1, 0]) + '+'
                    else:
                        y = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
                    bins.append(np_regroup[i - 1, 0])
                    tmp.append(y)

                bins.append(X[col].min() - 0.1)

                result_data['interval'] = tmp
                result_data['flag_0'] = np_regroup[:, 2]
                result_data['flag_1'] = np_regroup[:, 1]
                bins.sort(reverse=False)
                print('Интервалы для переменной %s' % col)
                print(result_data)

            except Exception as e:
                print(e)

        return X, bins

        
        
        
        
class DiscretizeByDecisionTree():
    """
    Дискретизация с использованием деревьев решений заключается в использовании дерева решений 
    для определения оптимальных точек разделения, которые определяют интервалы или непрерывные интервалы:
        
    1. Обучение дерева решений ограниченной глубины (2, 3 или 4) с использованием переменной, 
       которую мы хотим дискретизировать, чтобы предсказать целевую переменную.
    2. Затем исходные значения переменной заменяются вероятностью, возвращаемой деревом.

    Параметры
    ----------
    col: str
      столбец для дискретизации
    max_depth: int или список int
      максимальная глубина дерева. Может быть целым числом или списком целых чисел, 
      для которых мы хотим, чтобы модель дерева искала оптимальную глубину.
    
    """

    def __init__(self, col=None, max_depth=None, tree_model=None):
        self.col = col
        self._dim = None
        self.max_depth = max_depth
        self.tree_model = tree_model


    def fit(self, X, y, **kwargs):
        """Подгоняет кодировщик под X и y.
        Параметры
        ----------
        X : массивоподобный, форма = [n_samples, n_features]
            Обучающие векторы, где n_samples - количество выборок,
            а n_features - количество функций.
        y : массивоподобный, форма = [n_samples]
            Целевые значения.
        Возвращает
        -------
        self : кодировщик
            Возвращает self.
        """

        self._dim = X.shape[1]

        _, tree = self.discretize(
            X_in=X,
            y=y,
            max_depth=self.max_depth,
            col=self.col,
            tree_model=self.tree_model
        )
        self.tree_model = tree
        return self

    def transform(self, X):
        """Выполняет преобразование в новые категориальные данные.
        Будет использовать модель дерева и список столбцов для дискретизации
        столбца.
        Параметры
        ----------
        X : массивоподобный, форма = [n_samples, n_features]
        Возвращает
        -------
        X : новый фрейм данных с дискретизированным новым столбцом.
        """

        if self._dim is None:
            raise ValueError('Необходимо обучить кодировщик, прежде чем его можно будет использовать для преобразования данных.')

        # Проверяем, что размер правильный
        if X.shape[1] != self._dim:
            raise ValueError('Неожиданный размер входной размерности %d, ожидалось %d' % (X.shape[1], self._dim,))

        X, _ = self.discretize(
            X_in=X,
            col=self.col,
            tree_model=self.tree_model
        )

        return X 


    def discretize(self, X_in, y=None, max_depth=None, tree_model=None, col=None):
        """
        Дискретизация переменной с использованием DecisionTreeClassifier

        """

        X = X_in.copy(deep=True)

        if tree_model is not None:  # Преобразование
            X[col+'_tree_discret'] = tree_model.predict_proba(X[col].to_frame())[:,1]

        else: # Обучение
            if isinstance(max_depth,int):
                tree_model = DecisionTreeClassifier(max_depth=max_depth)
                tree_model.fit(X[col].to_frame(), y)
                
            elif len(max_depth)>1:
                score_ls = [] # Здесь буду хранить roc auc
                score_std_ls = [] # Здесь буду хранить стандартное отклонение roc_auc
                for tree_depth in max_depth:
                    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
                    scores = cross_val_score(tree_model, X[col].to_frame(), y, cv=3, scoring='roc_auc')
                    score_ls.append(np.mean(scores))
                    score_std_ls.append(np.std(scores))
                temp = pd.concat([pd.Series(max_depth), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
                temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']
                max_roc = temp.roc_auc_mean.max()
                optimal_depth=temp[temp.roc_auc_mean==max_roc]['depth'].values
                tree_model = DecisionTreeClassifier(max_depth=optimal_depth)
                tree_model.fit(X[col].to_frame(), y)

            else:
                raise ValueError('max_depth дерева должен быть целым числом или списком целых чисел')

        return X, tree_model


