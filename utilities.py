import pandas as pd
import numpy as np
from math import *
from scipy import stats as st
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score


def common_idxs_list(data_1, data_2):
    """
    Funzione che ritorna la lista
    degli indici comuni negli oggetti
    passati
    
    Parametri
    ---------
    
    data_1: Formato pandas.DataFrame,
            pandas.Series
            primo blocco di dati
            
    data_2: Formato pandas.DataFrame,
            pandas.Series
            secondo blocco di dati
            
    Valore ritornato
    ----------------
    
    Lista degli indici comuni
    
    """
    return list(data_1.index & data_2.index)





def common_idxs_objs(data_1, data_2):
    """
    Funzione che ritorna gli oggetti
    passati, per gli indici comuni
    
    Parametri
    ---------
    
    data_1: Formato pandas.DataFrame, 
            pandas.Series
            primo blocco di dati
            
    data_2: Formato pandas.DataFrame,
            pandas.Series
            secondo blocco di dati
            
    Valore ritornato
    ----------------
    
    I due oggetti, formati dai soli elementi
    aventi indice comune con quelli dell'altro
    oggetto
    
    """
    return data_1.loc[list(data_1.index & data_2.index)], \
           data_2.loc[list(data_1.index & data_2.index)]
    
    
    
    
    
def check_norm(data):
    """
    Funzione che controlla se le colonne
    di una matrice di dati sono normali
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame, 
          pandas.Series
    """
    
    data = data.copy()
    for colonna in data:
        ris = st.normaltest(data[colonna])
        if ris[1] > 0.05:
            print('{}\nt_oss =  {:.2f}, pvalue = {:.2f}'\
                  ', accetto normalità \n----------'. \
                  format(colonna, ris[0], ris[1]))
        else:
            print('{}\nt_oss =  {:.2f}, pvalue = {:.2f}'\
                  ', rifiuto normalità \n----------'. \
                  format(colonna, ris[0], ris[1]))
    
    





def pattern_consumi_mensili(data, cons_dict):
    """
    Funzione che ritorna il DataFrame
    del pattern dei consumi
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame, 
          pandas.Series
          
    cons_dict: Formato dict
               dizionario dei consumi
               
    Valore ritornato
    ----------------
    
    DataFrame del pattern dei consumi
    """
    
    mesi = []
    for key, item in cons_dict.items():
        mesi.extend(item.index.month.tolist())
    mesi = np.unique(mesi)
    df = pd.DataFrame(dict(zip(mesi, [np.nan]*len(data))), index=data.index)

    for key, item in cons_dict.items():
        medie_mensili = item.groupby([item.index.month]).mean().tolist()
        df.loc[key, mesi] = medie_mensili
        
    return df









def pattern_consumi_mensili_bis(data, cons_dict):
    """
    Funzione che ritorna il DataFrame
    del pattern dei consumi
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame, 
          pandas.Series
          
    cons_dict: Formato dict
               dizionario dei consumi
               
    Valore ritornato
    ----------------
    
    DataFrame del pattern dei consumi
    """
    
    anni = []
    for key, item in cons_dict.items():
        anni.extend(item.index.month.tolist())
    df = pd.DataFrame(dict(zip(np.unique(anni).tolist(), [np.nan]*len(data))), index=data.index)

    for key, item in cons_dict.items():
        media_complessiva = np.mean(item)
        medie_annue = item.groupby([item.index.month]).mean().tolist()
        mnth = np.unique(item.index.month)#[1:]
        temp = [0]
        for i in range(1, len(medie_annue)):
            confronto = medie_annue[i] - medie_annue[i-1]
            # constante
            if 0 <= confronto <= 10:
                temp.append(1)
            # spike
            elif confronto > 10:
                temp.append(2)
            # drop
            else:
                temp.append(0)
        df.loc[key, mnth] = temp
        
    return df.fillna()






def best_gmm(X, n_comp):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_comp)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm



    
    
def gmm_comp_selection_bic(data):
    max_components = data.shape[1]
    min_crit = np.Infinity
    best_n_comp = None
    for n_comp in range(1, max_components + 1):
        gmm_estimator = GaussianMixture(n_components=n_comp).fit(data)
        crit = gmm_estimator.bic(data)
        if crit < min_crit:
            min_crit = crit
            best_n_comp = n_comp
    return best_n_comp






def misure_bonta(lista):
    """
    Funzione che ritorna le misure di bontà
    tpr, fpr, tnr, fnr, accuracy, precision,
    recall
    
    
    Parametri
    ---------
             
    lista = tp, fp, fn, tn: formato int
                            sono il numero, rispettivamente,
                            di veri positivi, falsi positivi, 
                            falsi negativi e veri negativi
             
    Valore ritornato
    ----------------
    Le misure di bontà
    tpr, fpr, tnr, fnr, accuracy, precision,
    recall

    """ 
    tp, fp, fn, tn = lista
    fpr = fp / (tn + fp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fnr = fn / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    lift = (tp / (tp + fn)) / ((tp + fp) / (tp + tn + fp + fn))
    return np.array([tpr, fpr, fnr, tnr, accuracy, precision, recall, lift])

