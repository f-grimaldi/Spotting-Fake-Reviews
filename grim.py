import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import *
from scipy import stats
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import tree
from itertools import product
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture


def plot_roc_curve_grimaldi(y_true, y_pred, title_fontsize=22, 
                   text_fontsize=17, tick_fontsize=15, label=""):
    """
    Funzione che permette di rappresentare 
    la curva ROC 
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_pred: formato list
             contiene i valori predetti della y

    """ 
    plt.figure(figsize=(20,6))
    zipped = zip(y_true, y_pred)
    k = 0
    for i,j in zipped:
        fpr, tpr, _ = roc_curve(i, j)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='%s ROC (area = %0.2f)' % (label[k], area))
        k = k + 1
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random classification')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False positive rate", fontsize=text_fontsize)
    plt.ylabel("True positive rate", fontsize=text_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=text_fontsize, shadow=True)
    plt.title("ROC Curve", 
              fontsize=title_fontsize)



def be_ready(df):
    y_true = []
    y_true.append(df["Local_true"])
    y_true.append(df["tf-idf_true"])
    y_true.append(df["W2C_true"])

    y_pred = []
    y_pred.append(df["Local_pred"])
    y_pred.append(df["tf-idf_pred"])
    y_pred.append(df["W2C_pred"])

    return y_true, y_pred