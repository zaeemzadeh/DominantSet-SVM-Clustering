import numpy as np
from numpy.linalg import norm

from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
import sklearn.svm as svm
import matplotlib.pyplot as plt


def compute_kernel(X, Y=None, metric='mahalanobis'):
    D = pairwise_distances(X, Y, metric=metric)
    sigma = np.median(D)
    D /= sigma
    D == D**2
    S = np.exp(-D)
    return S


def dominant_set(A, x=None, epsilon=1.0e-4):
    """Compute the dominant set of the similarity matrix A with the
    replicator dynamics optimization approach. Convergence is reached
    when x changes less than epsilon.

    See: 'Dominant Sets and Pairwise Clustering', by Massimiliano
    Pavan and Marcello Pelillo, PAMI 2007.
    """
    if x is None:
        x = np.ones(A.shape[0]) / float(A.shape[0])

    distance = epsilon * 2
    while distance > epsilon:
        x_old = x.copy()
        # x = x * np.dot(A, x) # this works only for dense A
        x = x * A.dot(x)  # this works both for dense and sparse A
        x = x / x.sum()
        distance = norm(x - x_old)
        # print x.size, distance

    return x


def ds_svm_clustering(X, n_clust=2, eta=2, ds_eps=2e-3, plot=False, metric='mahalanobis'):
    """Dominant set + SVM Clustering:
    Alg 1 in Unsupervised Action Discovery and Localization in Videos
    http://crcv.ucf.edu/papers/iccv17/Soomro_ICCV17.pdf
    Written by: Alireza Zaeemzadeh
    """
    if plot:
        plt.figure()
        for yi in np.unique(y):
            plt.plot(X[y==yi,0], X[y==yi,1], 'o')

        plt.title('Dataset')

    S = compute_kernel(X, metric=metric)

    spectral = SpectralClustering(n_clusters=n_clust, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(S)
    labels = spectral.labels_

    if plot:
        plt.figure()
        for l in np.unique(labels):
            plt.plot(X[labels == l, 0], X[labels == l, 1], 'o')

        plt.title('Spectral Clustering')

    # finding dominant sets
    for l in np.unique(labels):
        idx_l = np.where(labels == l)
        X_l = X[labels == l, :]

        S_l = compute_kernel(X_l, metric=metric)

        x = dominant_set(S_l, epsilon=ds_eps)

        cutoff = np.median(x[x > 0])
        dom_idx = x > cutoff

        for i in idx_l[0][~dom_idx]:
            labels[i] = -1
    if plot:
        plt.figure()
        for yi in np.unique(labels):
            plt.plot(X[labels == yi, 0], X[labels == yi, 1], 'o')

        plt.title('Dominant set of each cluster (from spectral clustering)')

    dominant_set_idx = np.where(labels != -1)[0].tolist()
    non_dominant_set_idx = np.where(labels == -1)[0].tolist()
    while len(non_dominant_set_idx) > 0:
        remove_idx = []

        for l in np.unique(labels):

            if l == -1:
                continue

            if len(non_dominant_set_idx) == 0:
                break

            n_select = eta if len(non_dominant_set_idx) >= eta else len(non_dominant_set_idx)

            ovr_classes = [1 if labels[dominant_set_idx[i]] == l else -1 for i in range(len(dominant_set_idx))]

            clf = svm.SVC( kernel='precomputed', tol=1e-5)


            S = compute_kernel([X[i, :] for i in dominant_set_idx], metric=metric)

            clf.fit(S, ovr_classes)

            S = compute_kernel([X[i, :] for i in non_dominant_set_idx], [X[i, :] for i in dominant_set_idx],
                                   metric=metric)
            scores = clf.decision_function(S)

            new_idx_l = np.argsort(scores)[-n_select:]
            remove_idx.extend(new_idx_l)

            for i in new_idx_l:
                dominant_set_idx.append(non_dominant_set_idx[i])
                labels[non_dominant_set_idx[i]] = l

            for idx in sorted(remove_idx, reverse=True):
                # print idx
                del non_dominant_set_idx[idx]
            remove_idx = []

    if plot:
        plt.figure()
        for yi in np.unique(labels):
            plt.plot(X[labels == yi, 0], X[labels == yi, 1], 'o')

        plt.title('Dominant set + SVM Clustering')

    return labels


if __name__ == '__main__':
    np.random.seed(6)

    nclust = 3
    N = 1000    # number of samples
    d = 2     # dimension of samples (number of features)
    weights = np.ones(nclust)
    weights /= sum(weights)

    X, y = make_classification(weights=weights.tolist(), n_classes=nclust, n_samples=N, n_features=d,
                               n_redundant=0, class_sep=1, n_clusters_per_class=1, n_informative=d)

    dist_metric = 'mahalanobis'  #cosine, euclidean, l1, l2, manhattan, mahalanobis
    labels = ds_svm_clustering(X, n_clust=nclust, plot=True, metric=dist_metric)
    print 'Adjusted Mutual Information Score: ', adjusted_mutual_info_score(y, labels)
    plt.show()