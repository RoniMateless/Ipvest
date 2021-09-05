import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from scipy.stats import mode
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score,confusion_matrix, silhouette_score, silhouette_samples, f1_score, davies_bouldin_score
from sklearn.metrics.cluster import v_measure_score, homogeneity_completeness_v_measure

import random
from generate_data import generate_features
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster
import time
from collections import defaultdict
from datetime import datetime
import matplotlib.dates as mdates
import collections
import operator
from utility import fancy_dendrogram, plot_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def init_clustering_objects(X, n_clusters):
    # ============
    # Create cluster objects
    # ============
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    connectivity = kneighbors_graph(X, n_neighbors=20, include_self=False)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    dbscan = cluster.DBSCAN(eps=0.3)
    affinity_propagation = cluster.AffinityPropagation()
    average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=n_clusters, connectivity=connectivity)
    birch = cluster.Birch(n_clusters=n_clusters)
    ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', affinity='euclidean')#,connectivity=connectivity)
    spectral = cluster.SpectralClustering(assign_labels="discretize", n_clusters=n_clusters)
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    kmeans_plus = cluster.KMeans(n_clusters=n_clusters, algorithm='full')
    #clustering_algorithms = (('Ward', ward),)
    clustering_algorithms = (('Ward', ward),
#                             ('MiniBatchKMeans', two_means),
     #                        ('AffinityPropagation', affinity_propagation),
     #                        ('MeanShift', ms),
     #                        ('SpectralClustering', spectral),
#                             ('AgglomerativeClustering', average_linkage),
     #                        ('DBSCAN', dbscan),
     #                        ('Birch', birch),
#                             ('KMeans', kmeans_plus),
       #                      ('GaussianMixture', gmm),
                             )
    return clustering_algorithms


def plot_clustering_scores(min_n_clusters, max_n_clusters, km_scores, vmeasure_score, km_silhouette, db_score, gm_bic):

    plt.figure(figsize=(7,4))
    #plt.title("The elbow method",fontsize=16)
    plt.scatter(x=[i for i in range(min_n_clusters,max_n_clusters)],y=km_scores,s=150,edgecolor='k')
    #plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=12)
    plt.ylabel("GMM score",fontsize=12)
    plt.xticks([i for i in range(min_n_clusters,max_n_clusters)],fontsize=14)
    plt.yticks(fontsize=12)
    #plt.show()
    fig = "fig-servers-lin-elbow.png"
    plt.savefig(fig, bbox_inches='tight', dpi=100)

    plt.figure(figsize=(7,4))
    plt.title("The V-measure score",fontsize=16)
    plt.scatter(x=[i for i in range(min_n_clusters,max_n_clusters)],y=vmeasure_score,s=150,edgecolor='k')
    #plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("V-measure score",fontsize=15)
    plt.xticks([i for i in range(min_n_clusters,max_n_clusters)],fontsize=14)
    plt.yticks(fontsize=12)
    #plt.show()

    plt.figure(figsize=(7,4))
    #plt.title("The silhouette coefficient method",fontsize=16)
    plt.scatter(x=[i for i in range(min_n_clusters,max_n_clusters)],y=km_silhouette,s=150,edgecolor='k')
    #plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=12)
    plt.ylabel("Silhouette score",fontsize=12)
    plt.xticks([i for i in range(min_n_clusters,max_n_clusters)],fontsize=14)
    plt.yticks(fontsize=12)
    #plt.show()
    fig = "fig-servers-lin-silhouette.png"
    plt.savefig(fig, bbox_inches='tight', dpi=100)

    #plt.scatter(x=[i for i in range(min_n_clusters,max_n_clusters)],y=db_score,s=150,edgecolor='k')
    #plt.grid(True)
    #plt.xlabel("Davies-Bouldin score")
    #plt.show()

    plt.figure(figsize=(7, 4))
    #plt.title("The GMM model BIC", fontsize=16)
    plt.scatter(x=[i for i in range(min_n_clusters,max_n_clusters)], y=np.log(gm_bic), s=150, edgecolor='k')
    #plt.grid(True)
    plt.xlabel("Number of clusters", fontsize=12)
    plt.ylabel("Log of GMM BIC score", fontsize=12)
    plt.xticks([i for i in range(min_n_clusters,max_n_clusters)], fontsize=14)
    plt.yticks(fontsize=12)
    #plt.show()
    fig = "fig-servers-lin-bic.png"
    plt.savefig(fig, bbox_inches='tight', dpi=100)


def get_matched_clustering_labels(cluster_labels, gt):
    labels = np.zeros_like(cluster_labels)
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    dict = {}
    for i in unique_clusters:
        mask = (cluster_labels == i)

        count = sum(1 for i in mask if i == True)
        m = mode(gt[mask])[0][0]
        count_mode = sum(1 for i in gt[mask] if i == m)
        dict[str(i)] = (count, count_mode/float(count))

        labels[mask] = mode(gt[mask])[0]
    return labels, dict

from scipy.misc import comb
def rand_index_score(clusters, classes, beta1, beta2, beta3):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    precision = (tp)/(tp + fp)
    recall = (tp)/(tp + fn)
    ri = (tp + tn) / (tp + fp + fn + tn)
    f1 = ((beta1 * beta1 + 1) * precision * recall) / (beta1 * beta1 * precision + recall)
    f2 = ((beta2 * beta2 + 1) * precision * recall) / (beta2 * beta2 * precision + recall)
    f3 = ((beta3 * beta3 + 1) * precision * recall) / (beta3 * beta3 * precision + recall)
    return ri, precision, recall, f1, f2, f3




from sklearn import mixture
def make_clustering(X, gt, start_device_index, n_clusters, Id2IP, OS_name, features_name, file_writer):
    X.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    print("Data for Clustering X: {}".format(X.shape))

    #Z = linkage(X, 'ward')
    #plt.figure(figsize=(8,8))
    #fancy_dendrogram(Z,truncate_mode='lastp',p=30,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,annotate_above=40,max_d=0.5,)
    #plt.show()

    min_n_clusters = 8#n_clusters#8#4#3#38
    max_n_clusters = 9#n_clusters+1#9#5#12#128


    for n_clusters in range (min_n_clusters, max_n_clusters):#[n_clusters]

        clustering_algorithms = init_clustering_objects(X, n_clusters)

        for alg_name, algorithm in clustering_algorithms:

            purify_l = []
            homogeneity_l = []
            completeness_l = []
            v_measure_1_l = []
            v_measure_05_l = []
            v_measure_02_l = []
            ars_l = []
            f_1_l = []
            precision_l = []
            recall_l = []
            f_05_l = []
            f_02_l = []
            ri_l = []
            cm_l = []
            time_l = []

            for i in range(20):
                t0 = time.time()

                algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)
                #print ("Time:", ('%.2fs' % (t1 - t0)).lstrip('0'))


                s = pd.Series(data=y_pred + start_device_index, index=X.index.get_values())

                unique, counts = np.unique(y_pred, return_counts=True)
                #print (dict(zip(unique, counts)))

                # match between cluster labels and ground truth
                labels, cluster2gt_score = get_matched_clustering_labels(y_pred, gt)
                acc = accuracy_score(gt, labels)
                #f1 =  f1_score(gt, labels, average='weighted')
                ars = adjusted_rand_score(gt, y_pred)
                ri, precision, recall, f_1, f_05, f_02 = rand_index_score(y_pred, gt, 1, 0.5, 0.2)

                cm = contingency_matrix(gt, y_pred)
                homogeneity, completeness, v_measure_1 = homogeneity_completeness_v_measure(gt, y_pred)
                v_measure_05 = v_measure_score(gt, y_pred,beta=0.5)
                v_measure_02 = v_measure_score(gt, y_pred, beta=0.2)
                v_measure_1_l.append(v_measure_1)
                v_measure_05_l.append(v_measure_05)
                v_measure_02_l.append(v_measure_02)
                purify_l.append(acc)
                cm_l.append(cm)
                #f1_l.append(f1)
                homogeneity_l.append(homogeneity)
                completeness_l.append(completeness)
                #v_measure_l.append(v_measure)
                ars_l.append(ars)
                ri_l.append(ri)
                precision_l.append(precision)
                recall_l.append(recall)
                f_1_l.append(f_1)
                f_05_l.append(f_05)
                f_02_l.append(f_02)
                time_l.append(t1-t0)

            #print (row_measures)

            #mat = confusion_matrix(gt, labels)
            #print (mat)

            #row_measures = [OS_name ,features_name, alg_name, n_clusters, np.average(acc_list), np.average(f1_list), np.average(homogeneity_list), np.average(completeness_list), np.average(vmeasure_score),
            #                np.average(km_scores), np.average(km_silhouette), np.average(db_score), np.average(gm_bic),np.average(time_list)]

            row_measures = [OS_name, features_name, alg_name, n_clusters, np.average(purify_l), np.average(ars_l), np.average(precision_l), np.average(recall_l), np.average(f_1_l), np.average(f_05_l), np.average(f_02_l), np.average(ri_l), np.average(homogeneity_l),
                            np.average(completeness_l), np.average(v_measure_1_l), np.average(v_measure_05_l), np.average(v_measure_02_l), np.average(time_l)]
            print (row_measures)
            file_writer.writerow(row_measures)

        #sorted_counts = sorted(cluster2gt_score.items(), key=operator.itemgetter(1), reverse=True)
        #print (sorted_counts)

        #num_groups_to_report = int(num_of_clusters / 3)
        #count=0
        #score=0
        #for c, v in sorted_counts[:num_groups_to_report]:
        #    count += v[0]
        #    score += v[1]*v[0]
        #tot_score = score/float(count)
        #print ("LargestGroups. Total Reported count:", count, "Out of", len(y_pred), ". Num of Groups:", num_groups_to_report, "Out of", num_of_clusters, "Clusters" , ", Total Score:", tot_score)

    return s


def set_number_of_groups(X, gt):
    X.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    print("Data for Clustering X: {}".format(X.shape))

    avg_km_scores = []
    avg_km_silhouette = []
    avg_vmeasure_score = []
    avg_db_score = []
    avg_gm_bic = []

    min_n_clusters = 3
    max_n_clusters = 12

    for n_clusters in range(min_n_clusters, max_n_clusters):
        gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
        alg_name, algorithm = 'GaussianMixture', gmm

        km_scores, km_silhouette, vmeasure_score, db_score, gm_bic = ([] for i in range(5))

        for i in range(20):
            algorithm.fit(X)

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            try:
                km_scores.append(-algorithm.score(X))
            except:
                km_scores.append(0)
            silhouette = silhouette_score(X, y_pred)
            km_silhouette.append(silhouette)
            # print("Silhouette score for number of cluster(s) {}: {}".format(n_clusters, silhouette))

            db = davies_bouldin_score(X, y_pred)
            db_score.append(db)
            # print("Davies Bouldin score for number of cluster(s) {}: {}".format(n_clusters, db))

            v_measure = v_measure_score(gt, y_pred)
            vmeasure_score.append(v_measure)
            # print("V-measure score for number of cluster(s) {}: {}".format(n_clusters, v_measure))

            # print("BIC for number of cluster(s) {}: {}".format(n_clusters, algorithm.bic(X)))
            # print("-" * 100)
            try:
                gm_bic.append(algorithm.bic(X))
            except:
                gm_bic.append(0)


        avg_km_scores.append(np.average(km_scores))
        avg_km_silhouette.append(np.average(km_silhouette))
        avg_vmeasure_score.append(np.average(vmeasure_score))
        avg_db_score.append(np.average(db_score))
        avg_gm_bic.append(np.average(gm_bic))

    plot_clustering_scores(min_n_clusters, max_n_clusters, avg_km_scores, avg_vmeasure_score, avg_km_silhouette,avg_db_score, avg_gm_bic)