import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])

'''
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
'''

'''
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
'''




#show less in denodgram
'''
from sklearn.cluster import AgglomerativeClustering
# Affinity = {“euclidean”, “l1”, “l2”, “manhattan”,
# “cosine”}
# Linkage = {“ward”, “complete”, “average”}
Hclustering = AgglomerativeClustering(n_clusters=10,
 affinity=‘euclidean’, linkage=‘ward’)
Hclustering.fit(Cx)
ms = np.column_stack((ground_truth,Hclustering.labels_))
df = pd.DataFrame(ms,
 columns = [‘Ground truth’,’Clusters’])
pd.crosstab(df[‘Ground truth’], df[‘Clusters’],
 margins=True)
'''
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


# ============
# Create cluster objects
# ============
# estimate bandwidth for mean shift
# bandwidth = cluster.estimate_bandwidth(X, quantile=.3)
# ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
# two_means = cluster.MiniBatchKMeans(n_clusters=num_of_clusters)
#connectivity = kneighbors_graph(X, n_neighbors=20, include_self=False)
#ward = cluster.AgglomerativeClustering(n_clusters=num_of_clusters, linkage='ward', affinity='euclidean',
#                                       connectivity=connectivity)

# spectral = cluster.SpectralClustering(n_clusters=num_of_clusters, eigen_solver='arpack',affinity="nearest_neighbors")
# dbscan = cluster.DBSCAN(eps=.3)
# affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
# average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=num_of_clusters, connectivity=connectivity)
# birch = cluster.Birch(n_clusters=num_of_clusters)
# kmeans = cluster.KMeans(n_clusters=num_of_clusters, random_state=0)
# gmm = GaussianMixture(n_components=num_of_clusters, covariance_type='full')

clustering_algorithms = (
    # ('MiniBatchKMeans', two_means),
    # ('AffinityPropagation', affinity_propagation),
    # ('MeanShift', ms),
    # ('SpectralClustering', spectral),
    ('Ward', ward),
    # ('AgglomerativeClustering', average_linkage),
    # ('DBSCAN', dbscan),
    # ('Birch', birch),
    # ('KMeans', kmeans),
)