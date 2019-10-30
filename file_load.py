import pickle
import os


def pickle_dump(obj, path):
    """
    Funzione che permette di salvare un oggetto
    Python in formato pickle
    
    
    Parametri
    ---------
    
    obj: Formato qualsiasi
         dati da salvare
                  
    path: Formato str
          stringa contenente il percorso in cui si vuole
          salvare l'oggetto, con annesso nome del file
          es. miopc/Desktop/cartella/nomefile.pkl

    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
        
               
def pickle_load(file):
    """
    Funzione che permette di caricare un file
    in formato pickle 
    
    
    Parametri
    ---------
                  
    file: Formato str
          stringa contenente il percorso in cui è situato
          il file da caricare
          es. miopc/Desktop/cartella/nomefile.pkl

    """
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj
