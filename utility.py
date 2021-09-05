import matplotlib.pyplot as plt
import itertools as it
from scipy.cluster.hierarchy import dendrogram
import numpy as np
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=(5, 5))
    #plt.figure()
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(title)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)#, fontsize='xx-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_xticklabels(classes, rotation=(45), fontsize=10, va='bottom', ha='left')
    plt.xticks(tick_marks, classes, rotation=45)#, fontsize='x-large')
    plt.yticks(tick_marks, classes)#, fontsize='x-large')

    cm = np.around(cm, decimals=2)  # rounding to display in figure
    thresh = cm.max() / 2.
    #print (thresh)
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]:  # print values different than zero
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     verticalalignment='center',
                     fontsize=10,
                     color="black" if cm[i, j] > thresh else "black")

    # fig = plt.gcf()
    # fig.set_size_inches(25, 18, forward=True)
    plt.tight_layout()
    plt.ylabel('True Label')#, fontsize='x-large')
    plt.xlabel('Predicted Label')#, fontsize='x-large')
    plt.show()
    fig = "fig51.png"
    #plt.savefig(fig, bbox_inches='tight', dpi=100)



#from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Inconsistency-Method
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def get_num_clusters_using_elbow(distance):
    distance_rev = distance[::-1]
    idxs = np.arange(1, len(distance) + 1)
    plt.plot(idxs, distance_rev)

    acceleration = np.diff(distance, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("Num Of Clusters (According to elbow):", k)
    return k

import datetime
a =  datetime.datetime.fromtimestamp(1550160280.546561000).strftime('%Y-%m-%d %H:%M:%S.%f')
#datetime.datetime.fromtimestamp(1550160280.546561000).strftime('%c')
print (a)
request_body = {
    "mappings": {
        "connection": {
            "properties": {
                "start_time": {"type": "date"}
            }
        }
    }
}
from elasticsearch import Elasticsearch
ES_HOST = {"host" : "localhost", "port" : 9200}
#es = Elasticsearch(hosts=[ES_HOST])
#res = es.indices.create(index='index111', body=request_body)