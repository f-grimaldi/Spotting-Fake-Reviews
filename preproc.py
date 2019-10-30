import pandas as pd
import numpy as np
from math import *
from scipy import *



def mode_imputer(data, col_to_imp, col_imp):
    """
    Funzione che associa ai valori mancanti nella 
    colonna specificata il valore più probabile, 
    basandosi sulla Categoria e sulla funzione moda.
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati in ingresso
                  
    col_to_imp: Formato str
                colonna di 'data' nella quale sono presenti
                valori mancanti da imputare.
                               
    col_imp: Formato str
             colonna di 'data' tramite la quale la funzione
             andrà a fare l'effettiva imputazione del dato
             mancante.
             
                               
    Valore ritornato
    ----------------
                 
    La funzione ritorna un pandas.DataFrame
    con, al posto dei valori mancanti, dei valori 
    basati sulla moda condizionata a 'col_imp'
    """
    data = data.copy()
    idxs = data[data[col_to_imp].isnull()].index.tolist()
    cat_idx_dict = dict(zip(idxs, 
    						[data[col_imp].loc[i] for i in idxs]))
    for idx in cat_idx_dict:
        data.loc[idx, col_to_imp] = \
        (data[col_to_imp][data[col_imp] == 
        	  cat_idx_dict[idx]]).mode()[0]
    return data






def mean_imputer(data, col_to_imp, col_imp):
    """
    Funzione che associa ai valori mancanti nella 
    colonna specificata il valore più probabile, 
    basandosi sulla Categoria e sulla funzione media.
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati in ingresso
                  
    col_to_imp: Formato str
                colonna di 'data' nella quale sono presenti
                valori mancanti da imputare.
                               
    col_imp: Formato str
             colonna di 'data' tramite la quale la funzione
             andrà a fare l'effettiva imputazione del dato
             mancante.
             
                               
    Valore ritornato
    ----------------
                 
    La funzione ritorna un pandas.DataFrame
    con, al posto dei valori mancanti, dei valori 
    basati sulla media condizionata a 'col_imp'
    """
    data = data.copy()
    idxs = data[data[col_to_imp].isnull()].index.tolist()
    cat_idx_dict = dict(zip(idxs, 
    	                [data[col_imp].loc[i] for i in idxs]))
    for idx in cat_idx_dict:
        data.loc[idx, col_to_imp] = \
        np.nanmean(data[col_to_imp][data[col_imp] == cat_idx_dict[idx]])
    return data






def del_id_nodata_nolast12months(dizio_consumi):
    """
    Funzione che elimina i record per i quali
    non ci siano almeno 12 mesi di dati di consumo 
    raccolti
    
    
    Parametri
    ---------
    
    dizio_consumi: Formato Dict (il contenuto sono
                   pandas.Series
                   dizionario dei consumi per ID
             
                               
    Valore ritornato
    ----------------
                 
    La funzione ritorna il dizionario privato
    dei record vuoti, o che non hanno storia
    per gli ultimi 12 mesi.
    
    """
    return {idx: cons for idx, cons in dizio_consumi.items()
             if (cons is not None) and 
                (cons[-12:].notnull().sum() == 12)
            } 