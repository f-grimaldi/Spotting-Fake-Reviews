import pandas as pd
import numpy as np
from math import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline






def tree_param_selection(albero, param_grid, X_train, X_test, y_train, y_test):
    """
    Funzione che ritorna un dataframe contenente
    l'accuracy score associato ad ogni combinazione
    delle modalità delle griglie di parametri in ingresso
    
    
    Parametri
    ---------
    
    albero: Formato tree.DecisionTreeClassifier()
            oggetto albero
               
    param_grid: formato list
                contiene le griglie di valori per i parametri
                
    X_train: formato pandas.DataFrame
             insieme di allenamento
             
    X_test: formato pandas.DataFrame
            insieme di test
            
    y_train: formato pandas.core.series.Series
             insieme di allenamento per la risposta
             
    y_test: formato pandas.core.series.Series
            insieme di test per la risposta
            
    Valore ritornato
    ----------------
    
    Dataframe contenente le combinazioni di valori dei
    parametri con associato l'accuracy score

    """ 
    pipeline = Pipeline([ 
        ('tree', albero)
    ])
    risultati = []
    for params in param_grid:
        # settaggio dei parametri per lo stimatore
        pipeline.set_params(**params)
        # training del modello
        pipeline.fit(X_train, y_train)
        # previsione sull'insieme di test, dopo aver allenato sul train
        y_pred = pipeline.predict(X_test)
        params['accuracy_score'] = accuracy_score(y_test, y_pred)
        risultati.append(params)
    return pd.DataFrame(risultati).sort_values('accuracy_score', ascending=False)