import pandas as pd
import numpy as np
from math import *
from scipy import stats
from astropy.stats import median_absolute_deviation




def cons_medio_mod_colonna(data, cons_dict, colonna):
    """
    Funzione che restituisce la lista dei consumi
    medi per modalità della colonna selezionata
    
    Parametri
    ---------
                  
    data: Formato pandas.DataFrame
          matrice dei metadati

    cons_dict: Formato dict 
               dizionario dei consumi, contiene
               pandas.Series
                               
    colonna: Formato str
             colonna di 'data' ch si vuole riscalare
             
    Valore ritornato
    ----------------
    
    La lista dei consumi medi per modalità 
    della colonna selezionata.
    

    """
    mod_colonna = np.unique(data[colonna])
    data = data.copy()
    for modalita in mod_colonna:
        indici = list(data[data[colonna] == modalita].index & cons_dict.keys())
        media_tot = 0
        for idx in indici:
            media_tot += np.nanmean(cons_dict[idx])
        media_tot /= len(indici)
        data.loc[indici, colonna] = media_tot
    return data[colonna].tolist()   






def cons_med_unit_diviso_cons_medio_mod(data, cons_dict, colonna):
    """
    Funzione che restituisce la lista dei consumi
    medi per unità diviso i consumi medi per
    modalità della colonna selezionata
    
    Parametri
    ---------
                  
    data: Formato pandas.DataFrame
          matrice dei metadati

    cons_dict: Formato dict 
               dizionario dei consumi, contiene
               pandas.Series
                               
    colonna: Formato str
             colonna di 'data' ch si vuole riscalare
             
    Valore ritornato
    ----------------
    
    La lista dei consumi
    medi per unità diviso i consumi medi per
    modalità della colonna selezionata
    

    """
    array_medie_modalita = np.array(cons_medio_mod_colonna(data, cons_dict, colonna))
    array_medie_unita = np.array([np.nanmean(cons_dict[key]) for key in cons_dict])
    return (array_medie_unita / array_medie_modalita).tolist()








def classi_potenza(data):
    """
    Funzione che ritorna la colonna della
    PotenzaDisponibile riorganizzata in classi
    
    
    Parametri
    ---------

                                    
    data: Formato pandas.DataFrame
          matrice dei metadati 

          
    Valore Ritornato
    ----------------
    
    La funzione ritorna la colonna della
    PotenzaDisponibile riorganizzata in classi

    """
    data = data.copy()
    classes = [(0, range(10)), (1, range(10, 50)), 
               (2, range(50, 100)), (3, range(100, 500)), 
               (4, range(500, 1000)), (5, range(1000, 3001))]

    for i in data.index:
        for class_ in classes:
            if int(data.loc[i, 'PotenzaDisponibile']) in class_[1]:
                data.loc[i, 'PotenzaDisponibile'] = class_[0]
    return data['PotenzaDisponibile']





                         
def nansd(data, cons_dict, colonna):
    """
    Funzione che calcola standard deviation
    per ogni colonna selezionata
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict 
           dati di consumo
                                    
    data: Formato pandas.DataFrame
          matrice dei metadati 
    
    colonna: Formato str
             colonna alla quale ci si
             condiziona
          
    Valore Ritornato
    ----------------
    
    La funzione ritorna la colonna di standard
    deviation corrispontente alla colonna
    selezionata.

    """
    mod_colonna = np.unique(data[colonna])
    data = data.copy()
    data['temp'] = np.zeros(len(data))
    for modalita in mod_colonna:
        indici = list(data[data[colonna] == modalita].index & cons_dict.keys())
        somma_stdev = []
        for idx in indici:
            somma_stdev.extend(cons_dict[idx])
        data.loc[indici, 'temp'] = np.nanstd(somma_stdev)
    return data['temp'].tolist() 






def nansd_unita(cons_dict):
    """
    Funzione che calcola standard deviation
    per ogni osservazione
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict 
           dati di consumo
                                    
    Valore Ritornato
    ----------------
    
    La funzione ritorna la lista di standard
    deviations corrispondenti ad ogni unità.

    """
    stdevs = []
    for idx in cons_dict:
            stdevs.append(np.nanstd(cons_dict[idx]))
    return stdevs
            

    
    
    
    
    
def media_ultimi_12_diviso_media_totale_per_unita(data, cons_dict):
    """
    Funzione che calcola la media degli ultimi 12 mesi
    diviso la media totale dei consumi, per ogni id
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict 
           dati di consumo
                                    
    data: Formato pandas.DataFrame
          matrice dei metadati prima di
          modificare i valori da stringa
          a media di consumi, per alcune colonne
          
    Valore Ritornato
    ----------------
    
    La funzione ritorna le medie dei consumi degli ultimi 
    12 mesi divise per la media totale dei
    consumi.

    """
    data = data.copy()
    medie = []
    for i in data.index:
        # +0.001 perché ci sono delle forme 0/0
        medie.append(np.nanmean(cons_dict[i][-12:]) / (np.nanmean(cons_dict[i])+0.001))
    return medie  








def mad_ultimi_12_diviso_mad_totale_per_unita(data, cons_dict):
    """
    Funzione che calcola il mad degli ultimi 12 mesi
    diviso il mad dei consumi complessivi, per ogni id
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict 
           dati di consumo
                                    
    data: Formato pandas.DataFrame
          matrice dei metadati prima di
          modificare i valori da stringa
          a media di consumi, per alcune colonne
          
    Valore Ritornato
    ----------------
    
    La funzione ritorna il mad degli ultimi 12 mesi
    diviso il mad dei consumi complessivi, per ogni id.

    """
    data = data.copy()
    mad = []
    for i in data.index:
        mad.append(median_absolute_deviation(cons_dict[i][-12:], ignore_nan=True) \
                   / (median_absolute_deviation(cons_dict[i], ignore_nan=True)+0.001))
    return mad  

    
    
    
    
    
def consumi_mensili_su_cons_medio_tot(data, cons_dict):
    """
    Funzione che ritorna il DataFrame
    dei consumi mensili medi diviso il consumo
    medio totale per unità
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame, 
          pandas.Series
          
    cons_dict: Formato dict
               dizionario dei consumi
               
    Valore ritornato
    ----------------
    
    DataFrame dei consumi mensili medi 
    diviso il consumo medio totale per unità
    """
    
    mesi = []
    for key, item in cons_dict.items():
        mesi.extend(item.index.month.tolist())
    mesi = np.unique(mesi)
    df = pd.DataFrame(dict(zip(mesi, [np.nan]*len(data))), index=data.index)

    for key, item in cons_dict.items():
        medie_mensili = item.groupby([item.index.month]).mean().tolist()
        df.loc[key, mesi] = medie_mensili / (np.nanmean(item)+0.001)
        
    return df    
    
    
    
    
    
        
        
        
        
        