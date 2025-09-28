from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from utils import *


class SimpleGB(BaseEstimator):
    def __init__(self, max_depth, n_estimators, learning_rate):
        """
        Строит бустинг для бинарной классификации

        Параметры
        ----------
        max_depth : int
            Максимальная глубина дерева
        n_estimators : int
            Количество деревьев
        learning_rate : float
            Скорость обучения
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        
    def fit(self, X_data, y_data):
        """
        Обучает модель на данных X_data и y_data

        Параметры
        ----------
        X_data : numpy array
            Матрица признаков
        y_data : numpy array
            Вектор целевых значений 

        Возвращает
        ----------
        self: SimpleGB
            Обученная модель
        """
        curr_pred = np.ones_like(y_data) * 0.5
        for iter_num in range(self.n_estimators):
            # Считаем антиградиент логистической функции потерь по предсказниям в точке curr_pred
            antigrad = y_data - sigmoida(curr_pred)
            # Обучаем DecisionTreeRegressor предсказывать антиградиент
            algo = DecisionTreeRegressor(max_depth=self.max_depth, criterion="friedman_mse")
            algo.fit(X_data, antigrad)
            self.estimators.append(algo)
            # Обновляем в каждой точке
            curr_pred += self.learning_rate * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        """
        Предсказывает классы для данных X_data

        Параметры
        ----------
        X_data : numpy array
            Матрица признаков

        Возвращает
        ----------
        predictions : numpy array
            Вектор предсказанных классов
        """
        return self.predict_proba(X_data) >= 0.5
    
    def predict_proba(self, X_data):
        """
        Предсказывает вероятности классов для данных X_data

        Параметры
        ---------
        X_data : numpy array
            Матрица признаков

        Возвращает
        ----------
        probabilities : numpy array
            Вектор предсказанных вероятностей класса
        """
        # Предсказание на данных
        res = np.ones(X_data.shape[0]) * 0.5
        for estimator in self.estimators:
            # Нужно сложить все предсказания деревьев с весом self.tau
            res += self.learning_rate * estimator.predict(X_data)
        return sigmoida(res)
    
    
class MulticlassGB(BaseEstimator):
    def __init__(self, max_depth, n_estimators, learning_rate):
        """
        Строит бустинг для многоклассовой классификации

        Параметры
        ----------
        max_depth : int
            Максимальная глубина дерева
        n_estimators : int
            Количество деревьев
        learning_rate : float
            Скорость обучения
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        
    def fit(self, X_data, y_data):
        """
        Обучает модель на данных X_data и y_data

        Параметры
        ----------
        X_data : numpy array
            Матрица признаков
        y_data : numpy array
            Вектор целевых значений 

        Возвращает
        ----------
        self: MulticlassGB
            Обученная модель
        """
        self.n_classes = len(np.unique(y_data))
        
        for i in range(self.n_classes):
            y_data_tmp = (y_data == i).astype(int)
            algo = SimpleGB(self.max_depth, self.n_estimators, self.learning_rate)
            algo.fit(X_data, y_data_tmp)
            self.estimators.append(algo)
        return self
    
    def predict(self, X_data):
        """
        Предсказывает классы для данных X_data

        Параметры
        ----------
        X_data : numpy array
            Матрица признаков

        Возвращает
        ----------
        predictions : numpy array
            Вектор предсказанных классов
        """
        return np.argmax(self.predict_proba(X_data), axis=0)
    
    def predict_proba(self, X_data):
        """
        Предсказывает вероятности классов для данных X_data

        Параметры
        ---------
        X_data : numpy array
            Матрица признаков

        Возвращает
        ----------
        probabilities : numpy array
            Вектор предсказанных вероятностей классов
        """
        return softmax(np.stack([np.array(estimator.predict_proba(X_data)) for estimator in self.estimators]))