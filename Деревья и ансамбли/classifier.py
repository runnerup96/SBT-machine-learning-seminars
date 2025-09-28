import numpy as np
from criterion import gini, entropy
from utils import Node, sampling_proba, one_hot_encode

class CustomDecisionTreeClassifier:

    all_criterions = {'gini': gini, 'entropy': entropy}

    def __init__(self, criterion_name='gini', min_samples_leaf=1, min_samples_split=2,  max_depth=np.inf, n_classes=None):
        """
        Дерево решений (классификатор)

        Параметры:
        ----------
        criterion_name: {"gini", "entropy"}, default="gini"
            Критерий разделения. Поддерживаемые критерии:
            "gini" для Gini Impurity и "entropy" для информационного прироста
        min_samples_split: int, default=2
            Минимальное количество объектов, при котором будет происходить разделение узла
        min_samples_leaf: int, default=1
            Минимальное количество объектов, которые должны быть в листовом узле
        max_depth: int, default=np.inf
            Максимальная глубина дерева
        n_classes: int, default=None
            Количество классов для классификации, 
            если n_classes=None, то количество классов будет посчитано 
            по тренировочной выборке

        """
        assert criterion_name in self.all_criterions.keys(), \
        					'Критерий должен быть из набора: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None
        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Выполняет разделение предоставленного подмножества данных и целевых значений с 
        использованием выбранного признака и порогового значения
        
        Параметры
        ----------
        feature_index : int
            Индекс признака для разделения
        threshold : float
            Пороговое значение для выполнения разделения
        X_subset : np.array типа float размера (n_objects, n_features)
            Матрица признаков выборки
        y_subset : np.array типа float размера (n_objects, n_classes)
            one-hot-encoding представление меток выборки
        
        Возвращает
        -------
        (X_left, y_left) : кортеж np.array того же типа, что и входные X_subset и y_subset
            Часть входной выборки, где выбранный признак x^j < порогового значения threshold
        (X_right, y_right) : кортеж np.array того же типа, что и входные X_subset и y_subset
            Часть входной выборки, где выбранный признак x^j >= порогового значения threshold
        """
        mask = X_subset[:, feature_index] >= threshold
        
        X_right = X_subset[mask]
        y_right = y_subset[mask]
        
        X_left = X_subset[~mask]
        y_left = y_subset[~mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Разделяе целевые метки на два подмножества по указанному признаку и пороговому значению
        
        Параметры
        ----------
        feature_index : int
            Индекс признака для разделения
        threshold : float
            Пороговое значение для выполнения разделения
        X_subset : np.array типа float размера (n_objects, n_features)
            Матрица признаков выборки
        y_subset : np.array типа float размера (n_objects, n_classes)
            one-hot-encoding представление меток выборки
        
        Возвращает
        -------
        y_left : np.array типа float размера (n_objects, n_classes)
            Часть входной выборки, где выбранный признак x^j < порогового значения threshold
        y_right : np.array типа float размера (n_objects, n_classes)
            Часть входной выборки, где выбранный признак x^j >= порогового значения threshold
        """

        mask = X_subset[:, feature_index] >= threshold
        
        y_right = y_subset[mask]
        y_left = y_subset[~mask]
        
        return y_left, y_right
    


    def choose_best_split(self, X_subset, y_subset):
        """
        Жадно выбирает лучший признак и лучшее пороговое значение относительно выбранного критерия
        
        Параметры
        ----------
        X_subset : np.array типа float размера (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество
        y_subset : np.array типа float размера (n_objects, n_classes)
            Оne-hot-encoding представление меток выборки
        
        Возвращает
        -------
        feature_index : int
            Индекс признака для разделения
        threshold : float
            Пороговое значение для выполнения разделения
        """

        best_gain = 0.0
        best_feature_index = None
        best_threshold = None

        I, N = self.criterion(y_subset), len(y_subset)
        for feature_index, x in enumerate(X_subset.T):
            x = np.unique(x)
            for threshold in x:
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                I_L, N_L = self.criterion(y_left), len(y_left)
                I_R, N_R = self.criterion(y_right), len(y_right)
                gain = I - N_L/N * I_L - N_R/N * I_R
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
                
        return best_feature_index, best_threshold
    

    
    def make_tree(self, X_subset, y_subset, curr_depth):
        """
        Рекурсивно строит дерево
        
        Параметры
        ----------
        X_subset : np.array типа float размера (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество
        y_subset : np.array типа float размера (n_objects, n_classes)
            one-hot-encoding представление меток классов выборки
        curr_depth : int
        	текущая глубина дерева
        
        Возвращает
        -------
        root_node : экземпляр класса Node
            Корень дерева
        """

        # Если выполнен критерий отсановки или выборка стала однородной (<=> критерий == 0.0) создаем лист
        min_samples_leaf_stop = len(X_subset) < 2 * self.min_samples_leaf
        min_samples_split_stop = len(X_subset) < self.min_samples_split
        max_depth_stop = curr_depth >= self.max_depth
        criterion_stop = self.criterion(y_subset) == 0.0

        if min_samples_leaf_stop or min_samples_split_stop or max_depth_stop or criterion_stop:
            proba = sampling_proba(y_subset)
            value = np.argmax(proba)
            root_node = Node(None, value, proba)

        else:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)
            if feature_index is None: # Если ни одно из разбиений не приводит к улучшению критерия создаем лист
                proba = sampling_proba(y_subset)
                value = np.argmax(proba)
                root_node = Node(None, value, proba)
            else:
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                root_node = Node(feature_index, threshold)
                self.depth = max(self.depth, curr_depth+1)
                root_node.left_child = self.make_tree(X_left, y_left, curr_depth+1)
                root_node.right_child = self.make_tree(X_right, y_right, curr_depth+1)
            
        return root_node
    
    

    def fit(self, X, y):
        """
        Строит дерево решений 
        
        Параметры
        ----------
        X : np.array типа float с формой (n_objects, n_features)
            Матрица признаков, представляющая данные для обучения
        y : np.array типа int с формой (n_objects, 1)
            Вектор меток классов
        """

        assert len(y.shape) == 2 and len(y) == len(X), 'Вектор меток класса должен быть размера (n_objects, 1)'
        self.criterion = self.all_criterions[self.criterion_name]
        self.n_classes = self.n_classes if self.n_classes is not None else len(np.unique(y))
        y = one_hot_encode(self.n_classes, y)
        self.root = self.make_tree(X, y, curr_depth=0)
        
        
        
    def predict(self, X):
        """
        Предсказывает метку класса для каждого объекта входной выборки
        
        Параметры
        ----------
        X : np.array типа float размера (n_objects, n_features)
            Матрица признаков выборки, для которой должны быть предсказаны метки классов

        Возвращает
        -------
        y_predicted : np.array типа int с формой (n_objects, 1)
            Вектор меток классов
        """

        y_predicted = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self.root
            while node.feature_index is not None:
                node = node.right_child if x[node.feature_index] >= node.value else node.left_child
            y_predicted[i] = node.value
        
        return y_predicted.reshape(-1, 1)

    
    
    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для каждого объекта входной выборки
        
        Параметры
        ----------
        X : np.array типа float размера (n_objects, n_features)
            Матрица признаков выборки, для которой должны быть предсказаны вероятности классов

        Возвращает
        -------
        y_predicted_probs : np.array типа float размера (n_objects, n_classes)
            Вероятности каждого класса для объектов входной выборки
        
        """

        y_predicted_probs = np.zeros((len(X), self.n_classes))
        for i, x in enumerate(X):
            node = self.root
            while node.feature_index is not None:
                node = node.right_child if x[node.feature_index] >= node.value else node.left_child
            y_predicted_probs[i] = node.proba
        return y_predicted_probs