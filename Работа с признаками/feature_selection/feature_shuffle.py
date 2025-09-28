import pandas as pd
#import numpy as np


from sklearn.ensemble import RandomForestClassifier #, RandomForestRegressor
from sklearn.metrics import roc_auc_score #, mean_squared_error



def feature_shuffle_rf(X_train,y_train,max_depth=None,class_weight=None, n_estimators=50,random_state=0):
    
    """
    Обнаружение признаков, перемешивание значений которых ведет к ухудшению целевой метрики
    
    Параметры
    ----------
    X_train : pd.DataFrame
        Входные данные в формате Pandas DataFrame.
    y_train : pd.Series
        Значение целевой переменной Pandas DataFrame.
    class_weight : float, optional
        Вес класса
    n_estimators: int
        Количество базовых алгоритмов в ансамбле
    random_state: int
        Seed значение
    

    Возвращает
    -------
    auc_drop: pd.Series
        Dataframe с признаками и изменением метрики auc_roc
    selected_features: pd.Series
        Dataframe с признаками у которых изменение auc-roc>0
    """
    
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, (model.predict_proba(X_train))[:, 1])
    feature_dict = {}

    # selection  logic
    for feature in X_train.columns:
        X_train_c = X_train.copy().reset_index(drop=True)
        y_train_c = y_train.copy().reset_index(drop=True)
        
        # перемешаем признак
        X_train_c[feature] = X_train_c[feature].sample(frac=1,random_state=random_state).reset_index(
            drop=True)
        #print(X_train_c.isnull().sum())
        # сделаем предсказание с перемешанным признаком и посчитаем auc-roc
        shuff_auc = roc_auc_score(y_train_c,
                                  (model.predict_proba(X_train_c))[:, 1])
        #print(shuff_auc)
        # сохраним разницу между исходным auc-roc и c перемешанным признаком
        feature_dict[feature] = (train_auc - shuff_auc)
        #print(feature_dict)
    
    auc_drop = pd.Series(feature_dict).reset_index()
    auc_drop.columns = ['feature', 'auc_drop']
    auc_drop.sort_values(by=['auc_drop'], ascending=False, inplace=True)
    selected_features = auc_drop[auc_drop.auc_drop>0]['feature']

    return auc_drop, selected_features

