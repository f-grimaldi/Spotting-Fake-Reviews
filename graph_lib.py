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
import graphviz
from sklearn.pipeline import Pipeline
from sklearn import tree
from itertools import product
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture


        
        

def plot_rel(cons_dict, data, colonna, rot=None):
    """
    Funzione che permette di disegnare la relazione
    tra consumo e colonna dei metadati
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict
               dati di consumo
           
    data: Formato pandas.DataFrame
          matrice dei metadati
          
    colonna: Formato str
             colonna di 'data' per la quale si 
             vogliono evidenziare i consumi
                               
    titolo: Formato str
            titolo del grafico
            
    rot: Formato str, es. 'vertical'
         rotazione delle etichette, default None

    """
    plt.figure(figsize=(20,6))
    cons_dict_means = [np.mean(cons_dict[i]) for i in cons_dict]
    df = pd.DataFrame({colonna:data[colonna], 'consumo': cons_dict_means})
    group_means = df.groupby(colonna).mean()
    ordered_x = group_means.index[np.argsort(group_means['consumo'])][::-1]
    ordered_y = np.sort(group_means['consumo'])[::-1]
    ax = sns.barplot(ordered_x, ordered_y, edgecolor="black", order=ordered_x,
                     palette=sns.cubehelix_palette(len(set(data[colonna])),
                                                   start=5, rot=-1, dark=0,
                                                   light=0.99, reverse=True))
    

    ax.set_ylabel('Consumo medio per \nogni modalità di {}'.format(colonna), fontsize=19)
    ax.set_xlabel(colonna, fontsize=19)
    plt.show()
    
    
    
    
    
    
def cons_medio_boxplots_col(cons_dict, data, colonna, rot=None):
    """
    Funzione che permette di disegnare i
    boxplots dei consumi delle modalità 
    della colonna selezionata
    
    
    Parametri
    ---------
    
    cons_dict: Formato dict
               dati di consumo
           
    data: Formato pandas.DataFrame
          matrice dei metadati
          
    colonna: Formato str
             colonna di 'data' per la quale si 
             vogliono evidenziare i consumi
                               
    titolo: Formato str
            titolo del grafico

    """
    plt.figure(figsize=(16,6))
    cons_dict_means = [log(cons_dict[i].mean()+1) for i in cons_dict]
    df = pd.DataFrame({colonna:data[colonna], 
                       'consumo': cons_dict_means},
                       index=data.index)

    group_medians = df.groupby(colonna).median()
    order = group_medians.sort_values(by=['consumo']).index[::-1]

    sns.boxplot(x=df[colonna], y=df['consumo'], orient='v',
                color='#5e9cea', linewidth=2, saturation=2,
                order=order)

    plt.xticks(rotation=rot)
    plt.ylabel('Log-consumo', fontsize=14)
    plt.xlabel(colonna, fontsize=14)
    plt.show()
    
    

    

def plot_pc_ist_curva(pc):
    """
    Funzione che disegna la curva della 
    varianza spiegata cumulata delle PC, 
    assieme al diagramma a barre della 
    percentuale di variabilità spiegata
    
    
    Parametri
    ---------
    
    pc: Formato sklearn.decomposition.pca.PCA
        componenti principali
           
    """
    plt.figure(figsize=(18, 7))
    exp_var_ratio = pc.explained_variance_ratio_
    
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.xlabel("Numero di componenti principali", fontsize=19)
    plt.ylabel("Percentuale di varianza\n spiegata cumulata",
               fontsize=19)
    plt.plot(exp_var_ratio.cumsum(), marker='o', ls='--', 
             color="black", lw=1.3)
    for i in range(len(exp_var_ratio)):
        plt.text(i - 0.5, exp_var_ratio.cumsum()[i] + 0.017, \
                 s=str(round(exp_var_ratio.cumsum()[i]*100, 2)) + "%", \
                 color="black", fontsize=9)
    ax = sns.barplot([i for i in range(len(exp_var_ratio))], exp_var_ratio, 
                     edgecolor="black", color='#5e9cea')
                     #palette=sns.cubehelix_palette(12, #12 gradazioni
                     #                              start=2, 
                     #                              rot=0, 
                     #                              dark=0.35, 
                     #                              light=2, 
                     #                              reverse=True))

    for i, j in zip(exp_var_ratio[1:], range(1, len(exp_var_ratio))):
        if j == 19:
            ax.text(j, 0.0007+0.006, '+'+str(0.07)+'%',
                horizontalalignment='center')
        else:
            ax.text(j, i+0.006, '+'+str(round(i, 4)*100)+'%',
                    horizontalalignment='center')
    plt.xticks(range(len(exp_var_ratio)), range(1, len(exp_var_ratio) + 1))
    plt.grid(linewidth=0.3);
        



def plot_pc_km_metadati(data, centers, n_clus, units_toprint):
    """
    Funzione che permette di rappresentare graficamente
    la prima componente principale verso le altre
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          matrice contenente i dati trasformati
          tramite minmaxscaler + pca
    
    n_clus: Formato int
            numero di cluster
    
    units_toprint: Formato int
                   unità da stampare
            
    """
    clus_labels = [int(i) for i in range(n_clus)]
    pc = [colonna for colonna in data if "clus" not in colonna]
    for i in pc[1:]:
        plt.figure(figsize=(13, 5))
        plt.xlabel("Componente {}".format(pc[0]), fontsize=14)
        plt.ylabel("Componente {}".format(i), fontsize=14)
        for cluster in clus_labels:
            x = data[data['cluster'] == cluster][pc[0]][:units_toprint]
            y = data[data['cluster'] == cluster][i][:units_toprint]
            plt.scatter(x, y, s=40, alpha=0.25, 
                        label='cluster {}'.format(cluster)
                        #,marker='$'+str(cluster)+'$'
                        )
            
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        
        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        
        plt.legend(loc=1, shadow=True)
        plt.show()
        
        
    
    
    
def plot_pc_gmm_metadati(data, centers, n_clus, units_toprint=2000):
    """
    Funzione che permette di rappresentare graficamente
    la prima componente principale verso le altre
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          matrice contenente i dati trasformati
          tramite minmaxscaler + pca
    
    n_clus: Formato int
            numero di cluster
    
    units_toprint: Formato int
                   unità da stampare
            
    """
    clus_labels = [int(i) for i in range(len(n_clus))]
    pc = [colonna for colonna in data if "clus" not in colonna]
    for i in pc[1:]:
        plt.figure(figsize=(13, 5))
        plt.xlabel("Componente {}".format(pc[0]), fontsize=14)
        plt.ylabel("Componente {}".format(i), fontsize=14)
        for cluster in clus_labels:
            x = data[data['cluster'] == cluster][pc[0]][:units_toprint]
            y = data[data['cluster'] == cluster][i][:units_toprint]
            plt.scatter(x, y, s=40, alpha=0.25, 
                        label='cluster {}'.format(cluster)
                        #,marker='$'+str(cluster)+'$'
                        )
            
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        
        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        plt.legend(loc=1, shadow=True)
        plt.show()
        

        
        
def plot_silhouette(data, max_clus, indici_casuali, comp_1, comp_2):
    """
    Funzione che permette di rappresentare graficamente
    la silhouette per numero di gruppi paria "k" e
    lo scatterplot di due componenti principali
    indicate, relativamente al numero di gruppi
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          matrice contenente i dati trasformati
          tramite minmaxscaler + pca
    
    max_clus: Formato int
              numero di cluster massimo
            
    indici casuali: Formato numpy.array
                    array contenente 10000
                    indici pescati casualmente
                    senza reinserimento
    
    comp_1, comp_2: Formato str
                    nome della colonna della componente 
                    principale da rappresentare
            
    """
    np.random.seed(seed=42)
    range_n_clusters = range(2, max_clus + 1)
    data = data.loc[indici_casuali, data.columns.values] 
    clusters_dimension = {}
    for n_clusters in range_n_clusters:
        
        # crea i gruppi
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(data)
        sil_score = silhouette_score(data, cluster_labels)
        centers = clusterer.cluster_centers_
        clusters_dimension[n_clusters] = cluster_labels

        # setta i subplot per silhouette e scatter delle PC
        # in ax1 e ax2 mi salvo i due grafici appaiati
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        sample_silhouette_values = silhouette_samples(data, 
        	                                          cluster_labels)
        y_lower = 10
        
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, 
                              alpha=0.5)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        # grafico silhouette
        #ax1.set_title("Grafico della silhouette per {} clusters" \
        #              .format(n_clusters),
        #              fontsize=19)
        ax1.set_xlabel("Valori della silhouette", fontsize=14)
        ax1.set_ylabel("Numero di clusters", fontsize=14)
        ax1.axvline(x=sil_score, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.annotate('Valore medio\ndella silhouette\npari a: {}' \
                     .format(round(sil_score, 4)), 
                     xy=(sil_score, 
                     	(len(data) + (n_clusters + 1) * 10)/2), 
                     xytext=(sil_score + 0.2, 
                            (len(data) + (n_clusters + 1) * 10)/4),
                             arrowprops=dict(facecolor='black', shrink=0.05))
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(data[comp_1], data[comp_2], marker='.',
                    s=80, lw=0, alpha=0.3,
                    c=colors, edgecolor='k')
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        #ax2.set_title("Componente {} vs componente {}" \
        #              .format(comp_1, comp_2),
        #              fontsize=19)
        ax2.set_xlabel("Componente {}".format(comp_1), fontsize=14)
        ax2.set_ylabel("Componente {}".format(comp_2), fontsize=14)
        #plt.suptitle(("For n_clusters = {} the average \
        #               silhouette_score is: {}" \
        #              .format(n_clusters, round(sil_score,3))),
        #             fontsize=14, fontweight='bold');        
    return clusters_dimension

          
        

def plot_id(dizio, ID):
    """
    Funzione che permette di rappresentare graficamente
    la curva di consumo di un preciso ID dato
    
    
    Parametri
    ---------
    
    dizio: Formato dict
               dizionario contenente i consumi
               
    ID: Formato int
        inspection ID della serie di consumi
            
    """
    plt.figure(figsize=(16,6))
    dizio[ID].plot(title="Consumo relativo all'ID {}".format(ID))
    plt.ylabel('KWh')
    plt.grid()
    plt.show()
    
    
    
def plot_id_pattern_consumi(ID, data):
    """
    Funzione che permette di rappresentare graficamente
    la curva di consumo di un preciso ID dato
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          df contenente i pattern dei consumi
               
    ID: Formato int
        inspection ID della serie di consumi
            
    """
    print(type(data.loc[ID]))
    data.loc[ID].plot(title="pattern consumo ID {}".format(ID))
    plt.xticks(data.columns.values, rotation='vertical')
    plt.yticks(np.arange(-1, 2), ['assenza', 'drop', 'spike'])
    plt.ylim(-1, 1.04)
    plt.axhline(y=0, color='red', linestyle='--')
    #plt.grid()
    plt.show()    
    
    

    
    
def plot_best_n_ids_for_cluster(new_data, original_names_and_feat_eng_df, cons_dict, n):
    """
    Funzione che permette di rappresentare graficamente
    le curve di consumo delle top n osservazioni
    per ogni cluster.
    Max 7 clusters
    
    
    Parametri
    ---------
    
    new_data: Formato pandas.dataFrame
              dataframe comprendente componenti
              principali, cluster, distanze
    
    
    original_names_and_feat_eng_df: Formato pandas.dataFrame
                                    dataframe prima delle componenti
                                    principali e del clustering, coi 
                                    nomi originali per categoria ecc,
                                    con le standard deviation

    
    cons_dict: Formato dict
               ad ogni ID corrispondono una
               serie di consumi
                
    
    n: Formato int
       numero di osservazioni da rappresentare
       per ogni cluster
            
    """
    inspids = new_data.index.values
    clusters = set(new_data['cluster'])
    colors = ['red', 'blue', 'green', 'darkorange']
    col_clus_dict = dict(zip(clusters, 
                             np.random.choice(colors, len(clusters), replace=False)))
    #plt.rcParams.update({'legend.labelspacing':0.75})
    for cluster, color in col_clus_dict.items():
        distanza_cluster_x = new_data['dist_clus_' + str(int(cluster))].values
        distanza_cluster_x_argsort = np.argsort(distanza_cluster_x)
        inspids_ordinati = inspids[distanza_cluster_x_argsort]
        oss_iesima = 0
        for inspid in inspids_ordinati[:n]:
            oss_iesima += 1
            plt.figure(figsize=(16, 5))
            plt.plot(cons_dict[inspid].index, cons_dict[inspid], c=color)
            plt.title("inspid {}".format(inspid), fontsize=14)
            plt.ylabel("Consumo per la {} unità più\n rappresentativa del cluster {}" \
                       .format(oss_iesima, int(cluster)), fontsize=17)
            plt.show()
            display(original_names_and_feat_eng_df.loc[[inspid]])
        
        
                      
            
            
            
            
            

    
    

    
    
def plot_tree_params(tree_params_df):
    """
    Funzione che permette di rappresentare in forma
    tabellare le combinazioni dei parametri cercati
    tramite cross validation per l'albero
    
    
    Parametri
    ---------
    
    tree_params_df: Formato pandas.dataFrame
                    dataframe composto dalle griglie
                    dei parametri, con associato l'accuracy
                    score
            
    """
    plt.figure(figsize=(13, 5))
    plt.scatter(tree_params_df['tree__max_depth'], 
                tree_params_df['tree__min_samples_leaf'], 
                c=tree_params_df['accuracy_score'], 
                cmap=plt.cm.viridis, alpha=0.75, s=400)
    plt.colorbar()
    plt.xlabel('Profondità massima', fontsize=14)
    plt.ylabel('Minimo unità per nodo foglia', fontsize=14)
    #plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    
    
def plot_tree(albero, nomi_features, nomi_classi, max_depth, out_file=None):
    """
    Funzione che permette di rappresentare 
    l'albero di classificazione
    
    
    Parametri
    ---------
    
    albero: Formato tree.DecisionTreeClassifier()
                   oggetto albero
    
    max_depth: formato int
               massima profondità dell'albero
               
    nomi_features: formato list
                   lista contenente i nomi delle colonne
                   
    nomi_classi: formato list
                 lista contenente le modalità di risposta 
                 fra le quali discriminare
    """     
    # ritorno dell'oggetto dot_data, ovvero l'albero fornito in 
    # ingresso rappresentato come stringa, in formato dot
    dot_data = tree.export_graphviz(
                    decision_tree=albero, 
                    out_file=out_file,
                    max_depth=max_depth,
                    feature_names=nomi_features,
                    class_names=nomi_classi,
                    filled=True, 
                    rounded=True,
                    special_characters=True
                )
    display(graphviz.Source(dot_data))
    
    
    
    

def plot_feature_importances(albero, nomi_features, max_num_features=50):
    """
    Funzione che permette di rappresentare 
    l'importanza delle colonne della matrice
    dei dati, sotto forma di istogramma
    
    
    Parametri
    ---------
    
    albero: Formato tree.DecisionTreeClassifier()
                   oggetto albero
               
    nomi_features: formato list
                   lista contenente i nomi delle colonne
                   
    max_num_features: formato int
                      numero massimo di feature rappresentabili
    """  
    plt.figure(figsize=(18, 6))
    feature_importances = pd.Series(albero.feature_importances_, 
                                    index=nomi_features).\
                                    sort_values(ascending=False)
    etichette = feature_importances.index
    valori_y = feature_importances.values
    #feature_importances[:max_num_features].plot(kind='bar', 
    #grid=True, title='Grado di importanza delle variabili')
    ax = sns.barplot(etichette, valori_y, 
                     edgecolor="black", 
                     color='#5e9cea')
                     #palette=sns.cubehelix_palette(len(etichette), #12 gradazioni
                     #                              start=2, 
                     #                              rot=0, 
                     #                              dark=0.35, 
                     #                              light=2, 
                     #                              reverse=True))
    plt.ylabel("Features \n importance", fontsize=19)
    #plt.xlabel('Features', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(lw=0.3)
    
    
    "Importanza delle features nel posizionare correttamente una unità all'interno di un cluster"
    
    
    
    
def plot_matrice_di_confusione(y_true, y_pred, classes, 
                               title="Confusion matrix", cmap=plt.cm.Blues,
                               threshold = 0.5):
    """
    Funzione che permette di rappresentare 
    la matrice di confusione
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_pred: formato list
            contiene i valori predetti della y
                   
    classes: formato list
             modalità di risposta
             
    title: formato str
           titolo del grafico
          
    cmap: formato matplotlib.cm
          colormap
    """  
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.04, pad=0.2)
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    #thresh = cm.max() * threshold
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), 
                 horizontalalignment="center", fontsize=18)
                 #,color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Real class")
    plt.xlabel("Predicted class")
    
    conf_matr_list = []
    for i in cm:
        for el in i:
            conf_matr_list.append(el)
    return conf_matr_list




def plot_roc_curve(y_true, y_pred, title_fontsize=22, 
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
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random classification')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False positive rate", fontsize=text_fontsize)
    plt.ylabel("True positive rate", fontsize=text_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=text_fontsize, shadow=True)
    plt.title("ROC Curve, area = {:.3f}".format(area), 
              fontsize=title_fontsize)

    
    
    
def plot_roc_curve_models(y_true, y_pred, leg='', title_fontsize=22, 
                          text_fontsize=17, tick_fontsize=15):
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
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=leg+',  AUC = {:.3f}'.format(area))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Tasso di Falsi Positivi", fontsize=text_fontsize)
    plt.ylabel("Tasso di Veri Positivi", fontsize=text_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=12, shadow=True)

    
    
    
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
    
    
    
    
def plot_precision_recall_curve(y_true, y_pred, 
                                title_fontsize=22, 
                                text_fontsize=17, 
                                tick_fontsize=15):
    """
    Funzione che permette di rappresentare 
    la Precision Recall Curve
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_pred: formato list
             contiene i valori predetti della y

    """ 
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    area = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label='Precision Recall Curve DecTree')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [y_true.mean(), y_true.mean()], 
             color='gray', lw=2, linestyle='--', label='Classificazione casuale')
    plt.xlabel("Recupero", fontsize=text_fontsize)
    plt.ylabel("Precisione", fontsize=text_fontsize)
    plt.tick_params(labelsize=text_fontsize)
    plt.legend(loc='best', fontsize=text_fontsize, shadow=True)
    plt.title("Precision Recall Curve, area = {:.3f}". \
              format(area), fontsize=title_fontsize)
    
    
    
    
def plot_precision_recall_curve_models(y_true, y_pred, leg='', 
                                title_fontsize=22, 
                                text_fontsize=17, 
                                tick_fontsize=15):
    """
    Funzione che permette di rappresentare 
    la Precision Recall Curve
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_pred: formato list
             contiene i valori predetti della y

    """ 
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    area = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label=leg+',  AUC = {:.3f}'. \
              format(area))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recupero", fontsize=text_fontsize)
    plt.ylabel("Precisione", fontsize=text_fontsize)
    plt.tick_params(labelsize=text_fontsize)
    plt.legend(loc='best', fontsize=12, shadow=True)
    
    

def plot_roc_rec_lift(y_test, y_score, title_fontsize=22, 
                      text_fontsize=19, tick_fontsize=15,
                      conf_matr_list=[], thresh=0.5):
    """
    Funzione che permette di rappresentare 
    la Curva ROC affiancata alla Precision Recall Curve
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_score: formato list
             contiene i valori predetti della y

    """ 
    plt.figure(figsize=(24, 8))
    plt.subplot(131)
    plot_roc_curve(y_test, y_score, title_fontsize=title_fontsize,
                   text_fontsize=text_fontsize, 
                   tick_fontsize=tick_fontsize)
    plt.grid()
    plt.subplot(132)
    plot_precision_recall_curve(y_test, y_score, 
                                title_fontsize=title_fontsize, 
                                text_fontsize=text_fontsize,
                                tick_fontsize=tick_fontsize)
    plt.grid()
    plt.subplot(133)
    plot_lift_curve_(y_test, y_score, class_names=['Fraud/Malf', 'Norm'], 
                    title_fontsize=title_fontsize, 
                    text_fontsize=text_fontsize, 
                    tick_fontsize=tick_fontsize,
                    conf_matr_list=conf_matr_list,
                    thresh=thresh)
    plt.grid()
    plt.tight_layout()
    plt.show()

    



    
    
def print_scores(y_test, y_pred):
    """
    Funzione che permette di stampare
    accuratezza, precisione e recupero
    
    
    Parametri
    ---------
    
    y_true: Formato list
            contiene i valori osservati della y
               
    y_pred: formato list
             contiene i valori predetti della y

    """ 
    # precisione nella classificazione
    print("Accuratezza {}: {:.2f}".format(leg, accuracy_score(y_test, y_pred)))
    
    # veri positivi / (veri positivi + falsi positivi)
    print("Precisione {}: {:.2f}".format(leg, precision_score(y_test, y_pred)))
    
    # veri positivi / (veri positivi + falsi negativi)
    print("Recupero {}: {:.2f}".format(leg, recall_score(y_test, y_pred)))
    
    

    
    
    
    
def plot_comp_importance(data, max_comps=1):
    plt.figure(figsize=(18, 6))
    plt.ylim(0, 1)
    gmm = GaussianMixture(n_components=max_comps).fit(data)
    print(gmm.weights_)
    plt.xticks(range(1, max_comps + 1))
    plt.bar(range(1, max_comps + 1), gmm.weights_)
    plt.show()
    

        
        
def scelta_covarianze_gmm_(data, max_comps=2, random_state=42):
    """
    Funzione che permette di stampare,
    per ogni struttura della matrice di
    covarianza, i gruppi formati dall'algoritmo
    con 2, 3, 4, 5 componenti.
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati post scaling e pca
               
    max_comps: formato int
               numero massimo di componenti 
               per l'algoritmo di clustering

    """ 
    if max_comps < 2:
        raise ValueError('max_comps deve essere > 2')
    cvs = ['full', 'tied', 'diag', 'spherical']
    gmm = []
    
    for comp in range(2, max_comps + 1):
        gmm.append(GaussianMixture(n_components = comp, random_state = random_state))
    
    for cv in cvs:
        kk = 100 + 10*max_comps + 1
        plt.subplots(figsize=(26,7))
        plt.suptitle('Struttura covarianza {} con 2, 3, 4 e 5 componenti'.\
                     format(cv),
                      ha='right', size=19, va='top')
        for model in gmm:
            
            plt.subplot(kk)
            kk += 1
            # imposto la covarianza
                   
            model.set_params(covariance_type=cv)
                   
            # fitto i dati
            model.fit(data)
                   
            # salvo predicts e medie
            predicts = model.predict(data)
            centers = model.means_
            
            x, y = centers[:,0], centers[:,1]

            
            plt.scatter(data['pc_1'], data['pc_2'], c=predicts)
            plt.scatter(x, y, marker='o', s=110, edgecolor='k', alpha=1, c='white')
            plt.xlabel('Componente pc_1', fontsize=14)
            plt.ylabel('Componente pc_2', fontsize=14)
            for j, c in enumerate(centers):
                plt.scatter(c[0], c[1], marker='$%d$' % j, alpha=1,
                                    s=50, edgecolor='k')
            
        plt.show()
    
    
    
    
    
def plot_cov_comp(data, cv_type='full', n_comp=2):
    """
    Funzione che permette di stampare i gruppi
    in corrispondenza della prima e seconda PC, 
    data la struttura della matrice di covarianza
    e il numero di componenti per l'algoritmo
    di clustering GaussianMixture
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati post scaling e pca
          
    cv_type: Formato str
             struttura della matrice di covarianza
               
    n_comp: formato int
            numero di componenti 
            per l'algoritmo di clustering

    """ 
    gmm = GaussianMixture(covariance_type=cv_type, n_components=n_comp,
                              random_state=42)
    gmm.fit(data)
    predicts = gmm.predict(data)
    centers = gmm.means_
    x, y = centers[:,0], centers[:,1]
    plt.scatter(data['pc_1'], data['pc_2'], c=predicts)
    plt.scatter(x, y, marker='o', s=110, edgecolor='k', alpha=1, c='white')
    plt.xlabel('Componente pc_1', fontsize=14)
    plt.ylabel('Componente pc_2', fontsize=14)
    for j, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % j, alpha=1,
                            s=50, edgecolor='k')
    
    
    
    
    


def plot_bar_fraudmalf_norm(data, y):
    """
    Funzione che permette di stampare il diagramma
    a barre di frodi, malfunzionamenti e casi
    normali.
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati post scaling, pca e gaussmix/kmeans
          
    y: Formato pandas.Series
       risultato delle ispezioni

    """ 
    y.name = 'risultato'
    yy = pd.get_dummies(y)
    cluster_risultato = data[['cluster']].join(yy)
    clusters = np.unique(data['cluster'])
    df = pd.DataFrame(index = clusters)
    for i in clusters:
        temp = []
        for j in yy:
            df.loc[i, j] = sum(cluster_risultato[cluster_risultato['cluster']==i][j])
    for i in df.index:
        somma_i = sum(df.loc[i])
        for j in df:
            df.loc[i, j] = df.loc[i, j] / somma_i

    gruppi, risultati, perc = [], [], []
    for i in clusters:
        for j in df:
            gruppi.append(i)
            risultati.append(j)
            perc.append(df.loc[i, j])

    dff = pd.DataFrame({'gruppi': gruppi,
                        'risultati': risultati, 
                        'perc': perc},
                        index=range(len(gruppi)))
    plt.figure(figsize=(18, 6))
    palette = sns.color_palette('hls', len(clusters))
    ax = sns.barplot(x='gruppi', y='perc', hue='risultati',
                data=dff, palette=palette, edgecolor="black")
    ax.set_xlabel('Gruppi', fontsize=14)
    ax.set_ylabel('Percentuale di frodi, malfunzionamenti\n' \
                  'e casi normali sul totale, per gruppo',
                  fontsize=14)
    xx = []
    for i in clusters:
        xx.extend(i+np.array([-0.3, 0, 0.3]))
    percs = []
    for i in dff['perc']:
        percs.append(str(round(i*100, 2))+'%')
    for i in range(len(percs)):
        ax.text(xx[i], dff['perc'][i]+0.006, percs[i],
                horizontalalignment='center')


        
        
        
        
        
def plot_dimensione_gruppi_bar(data):
    """
    Funzione che permette di stampare il diagramma
    a barre della dimensione dei gruppi
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati post scaling, pca e gaussmix/kmeans

    """ 
    plt.figure(figsize=(16,6))
    
    gmm_counts = np.unique(data['cluster'], return_counts=True)
    gruppi, sizes = gmm_counts[0], gmm_counts[1]
    
    labels = ['cluster {}'.format(int(i)) for i in gruppi]
    
    ax = sns.barplot(x=gruppi, y=sizes)
    
    for i, j in zip(sizes, range(len(sizes))):
            ax.text(j, i+500, str(i)+' unità',
                    horizontalalignment='center')
            
    plt.xlabel('Clusters', fontsize=14)
    plt.ylabel('Unità per cluster', fontsize=14)
    plt.show()
    
    
    
        
        
        
def plot_dimensione_gruppi_pie(data):
    """
    Funzione che permette di stampare il diagramma
    a barre della dimensione dei gruppi
    
    
    Parametri
    ---------
    
    data: Formato pandas.DataFrame
          dati post scaling, pca e gaussmix/kmeans

    """ 
    plt.figure(figsize=(7,7))
    gmm_counts = np.unique(data['cluster'], return_counts=True)
    gruppi, sizes = gmm_counts[0], gmm_counts[1]
    labels = ['cluster {}'.format(int(i)) for i in gruppi]
    explode = (0.1, 0.1, 0.1, 0.1) # esplode tutte le fette
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # così la torta viene circolare
    plt.legend(['Dimensione '+label+': '+str(dim)+' unità' \
                for label, dim in zip(labels, sizes)], 
                bbox_to_anchor=(0.5, 0.5, 0.5, 0.5), shadow=True)
    plt.show()
    
    
    
    
    
def boxplot_feature_per_cluster(data, data_clus):
    clusters = np.unique(data_clus['cluster'])
    
    for colonna in data:
        plt.subplots(figsize=(16, 6))
        y = []
        for cluster in clusters:
            y.append(data[data_clus['cluster']==cluster][colonna].values.reshape(-1, 1))
            # .values.reshape() perché altrimenti al prossimo 
            # aggiornamento 'la scrittura è deprecata'
            
        df = pd.DataFrame({colonna: data[colonna].astype('float'), 
                           'cluster': data_clus['cluster']},
                           index=data.index)
        group_medians = df.groupby('cluster').median()
        order = group_medians.sort_values(by=[colonna]).index[::-1]
        
        #plt.title('Distribuzione di {} nei cluster'.format(colonna),
        #          fontsize=17)
        
        sns.boxplot(x=clusters, y=y, orient='v',
                    color='#5e9cea', linewidth=2,
                    order=order,
                    saturation=2)
        
        plt.xlabel('Gruppi', fontsize=14)
        plt.ylabel('{}'.format(colonna), fontsize=14)
        labels = ['Mediana cluster {} = {}'.format(cluster, round(val[0], 2)) \
                  for cluster, val in zip(clusters, group_medians.values)]
        plt.legend(loc=1, labels=labels, shadow=True)#, framealpha=1)

        plt.show()
            
    

    
    
    
    
    
def cumulative_gain_curve(y_true, y_score):#, pos_label=None):
    """
    La funzione ritorna le coordinate per disegnare la
    curva lift, funziona solo per classi binarie
    
    Parametri
    ---------
    
    y_true: Formato np.array, valori 0 e 1
            y vere
            
    y_probas: Formato numpy.array
              valori predetti delle y, scores

    pos_label: Formato int, str, default=None

    Valore ritornato
    ----------------
    percentages (numpy.ndarray): ascisse del cumulative gain plot

    gains (numpy.ndarray): ordinate del cumulative gain plot
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    '''
    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError('dati non binari e pos_label non specificata')
    elif pos_label is None:
        pos_label = 1

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    '''
   
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)
    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains





def plot_lift_curve(y_true, y_probas, title='Lift Curve',
                    ax=None, figsize=None, title_fontsize=22,
                    text_fontsize=17, class_names=[], tick_fontsize=15,
                    conf_matr_list=[], thresh=0.5):
    """
    Funzione che disegna la curva lift legata
    alle y vere e agli scores calcolati
    
    y_true: Formato np.array, valori 0 e 1
            y vere
            
    y_probas: Formato numpy.array
              valori predetti delle y, scores
              
    title: Formato str
           titolo del grafico
           
    ax: Formato matplotlib.axes
        asse specificato
           
    figsize: Formato tuple    
             dimensioni finestra grafica
             
    title_fontsize: Formato float
                    dimensione titolo
    
    text_fontsize: Formato float
                   dimensione testo
    
    class_names: Formato list, tuple, contiene stringhe
                 nomi delle classi
    
    tick_fontsize: Formato float
                   dimensione ticks
           
           
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('impossibile calcolare la curva per dati con '
                         '{} categorie'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas,
                                                classes[0])
    #percentages, gains2 = cumulative_gain_curve(y_true, y_probas,
    #                                            classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    #gains2 = gains2[1:]

    gains1 = gains1 / percentages
    #gains2 = gains2 / percentages


    plt.title(title+', soglia = {}'.format(thresh), fontsize=title_fontsize)

    plt.plot(percentages, gains1, lw=2, label=class_names[0])
    #plt.plot(percentages, gains2, lw=3, label=class_names[1])

    plt.plot([0, 1], [1, 1], label='Classificazione casuale', color='gray', lw=2, linestyle='--')
    tp, fp, fn, tn = conf_matr_list
    lift = (tp / (tp + fn)) / ((tp + fp) / (tp + tn + fp + fn))
    plt.xlabel('Proporzione di campione', fontsize=text_fontsize)
    plt.ylabel('Lift = {}'.format(lift), fontsize=text_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=text_fontsize, shadow=True)
    #plt.show()

    
    
    
    
    
    
    
    
    
    
def plot_lift_curve_(y_true, y_probas, title='Lift Curve',
                    ax=None, figsize=None, title_fontsize=22,
                    text_fontsize=17, class_names=['Fraud/Malf'], tick_fontsize=15,
                    conf_matr_list=[], thresh=0.5):
    """
    Funzione che disegna la curva lift legata
    alle y vere e agli scores calcolati
    
    y_true: Formato np.array, valori 0 e 1
            y vere
            
    y_probas: Formato numpy.array
              valori predetti delle y, scores
              
    title: Formato str
           titolo del grafico
           
    ax: Formato matplotlib.axes
        asse specificato
           
    figsize: Formato tuple    
             dimensioni finestra grafica
             
    title_fontsize: Formato float
                    dimensione titolo
    
    text_fontsize: Formato float
                   dimensione testo
    
    class_names: Formato list, tuple, contiene stringhe
                 nomi delle classi
    
    tick_fontsize: Formato float
                   dimensione ticks
           
           
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('impossibile calcolare la curva per dati con '
                         '{} categorie'.format(len(classes)))
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas)
    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains1 = gains1 / percentages
    
    plt.plot(percentages, gains1, lw=2, label=class_names[0])
    plt.plot([0, 1], [1, 1], label='Classificazione casuale', color='gray', lw=2, linestyle='--')
    tp, fp, fn, tn = conf_matr_list
    lift = (tp / (tp + fn)) / ((tp + fp) / (tp + tn + fp + fn))
    plt.xlabel('Proporzione di campione', fontsize=text_fontsize)
    plt.ylabel('Guadagno cumulativo', fontsize=text_fontsize)
    plt.title(title+', soglia = {}, lift = {}'.format(thresh, round(lift, 3)), fontsize=title_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='best', fontsize=text_fontsize, shadow=True)
    #plt.show()
    
    
    
    
    
    
    
    
    
    
    
def plot_lift_curve_models(y_true, y_probas, leg='', lift=None,
                    ax=None, figsize=None, title_fontsize=22,
                    text_fontsize=17, class_names=[], tick_fontsize=15):
    """
    Funzione che disegna la curva lift legata
    alle y vere e agli scores calcolati
    
    y_true: Formato np.array, valori 0 e 1
            y vere
            
    y_probas: Formato numpy.array
              valori predetti delle y, scores
              
    title: Formato str
           titolo del grafico
           
    ax: Formato matplotlib.axes
        asse specificato
           
    figsize: Formato tuple    
             dimensioni finestra grafica
             
    title_fontsize: Formato float
                    dimensione titolo
    
    text_fontsize: Formato float
                   dimensione testo
    
    class_names: Formato list, tuple, contiene stringhe
                 nomi delle classi
    
    tick_fontsize: Formato float
                   dimensione ticks

    leg: Formato str
         stringhe di legenda aggiuntiva
         
    lift: Formato float
          valore del lift calcolato
           
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Lift Curve for data with '
                         '{} category/ies'.format(len(classes)))
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas)
    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains1 = gains1 / percentages
    lll = len(gains1)
    ten_to_forty_percs = [int(lll*p) for p in [0.1, 0.2, 0.3, 0.4]]
    #print(np.around(gains1, 2))
    #print(np.around(percentages, 2)*100)
    #plt.axvline(percentages[400])
    #plt.axhline(gains1[400])
    for i in ten_to_forty_percs:
    	plt.scatter(percentages[i], gains1[i])
    for i in ten_to_forty_percs:
    	print("{}% di campione per {} con guadagno cumulativo di {}".format(np.around(percentages[i], 2)*100, leg, np.around(gains1[i], 2)))
    plt.plot(percentages, gains1, lw=2, label=leg+' Fraud/Malf,  lift = {}'.format(lift))
    plt.xlabel('Proporzione di campione', fontsize=text_fontsize)
    plt.ylabel('Guadagno cumulativo', fontsize=text_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=12, shadow=True)
    #plt.show()