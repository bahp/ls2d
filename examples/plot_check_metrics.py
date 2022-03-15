"""
Compute GMM metrics
=========================

Sample file to test metrics.

"""





if __name__ == '__main__':

    # ------------------
    # Libraries
    # ------------------
    # Generic libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Basic functions in sklearn
    from sklearn import datasets
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.decomposition import PCA

    # Own functions
    from ls2d.utils import _load_pickle
    from ls2d.metrics import gmm_intersection_matrix
    from ls2d.metrics import gmm_ratio_score
    from ls2d.metrics import gmm_intersection_area
    from ls2d.metrics import create_ellipse_shapely
    from ls2d.metrics import create_ellipse_mpl


    def load_iris_embeddings_pca():
        """Compute iris PCA embeddings"""
        # Load iris dataset
        X, y = iris = datasets.load_iris(return_X_y=True)
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        # Compute transform
        y_pred = PCA(n_components=2).fit_transform(X)

        # Train pca and transform
        return X, y, y_pred


    def load_blob_embeddings():
        """Create 2D embeddings"""
        X, y = make_blobs(n_samples=500, centers=5,
                          cluster_std=2,
                          # centers=[[2,2], [1, 10], [10, 1]],
                          # cluster_std=[[1,1], [2,2], [0.5,0.5]],
                          n_features=2, random_state=170)
        return X, y, X

    # ------------------
    # Data
    # ------------------
    # Create iris samle
    #X, y, y_pred = load_iris_embeddings_pca()
    # Create blob sample
    X, y, y_pred = load_blob_embeddings()

    # Number of clusters
    n_clusters = len(np.unique(y))

    # ------------------
    # Metrics
    # ------------------
    """
    .. note: Overall it seems that the gmm_ratio_score is not very useful. 
             Note that minimising the ratios is the same as minimising the 
             intersection areas. The gmm_ratios matrix might be useful if 
             some classes are more important than others.
    """

    # Intersection are matrix
    gmm_matrix, label, gmms = \
        gmm_intersection_matrix(y_pred, y, include_all=False)
    # Intersection ratios
    gmm_ratios = gmm_matrix / np.diag(gmm_matrix)
    # Intersection normalization
    ratio_sum_v1 = np.sum(gmm_ratios) - np.sum(np.trace(gmm_ratios))
    # ratio_avg = ratio_sum /
    ratio_sum_v2 = gmm_ratio_score(y_pred, y, 'sum')

    # Show
    print("%s\n%s\n%s" % ('-'*50, 'Scores', '-'*50))
    print("\nAreas:")
    print(gmm_matrix)
    print("\nRatios:")
    print(gmm_ratios)
    print("\nRatio (sum v1):")
    print(ratio_sum_v1)
    print("\nRatio (sum v2):")
    print(ratio_sum_v2)

    # Compute scores
    scores = pd.Series(dtype='float')
    scores['gmm_intersection_area'] = \
        gmm_intersection_area(gmm_matrix=gmm_matrix)
    scores['gmm_intersection_area_norm'] = \
        gmm_intersection_area(gmm_matrix=gmm_matrix, normalize=True)
    scores['silhouette'] = metrics.silhouette_score(y_pred, y, metric="sqeuclidean")
    scores['calinski_h'] = metrics.calinski_harabasz_score(y_pred, y)
    scores['davies_bouldin'] = metrics.davies_bouldin_score(y_pred, y)
    scores['homogeneity'] = metrics.homogeneity_score(y, y)      # useless
    scores['completeness'] = metrics.completeness_score(y, y)    # useless
    scores['v_measure'] = metrics.v_measure_score(y, y)          # useless
    scores['adj_rs'] = metrics.adjusted_rand_score(y, y)         # useless
    scores['adj_mis'] = metrics.adjusted_mutual_info_score(y, y) # useless

    # Show
    print("\nScores:")
    print(scores)

    # ------------------
    # Show
    # ------------------
    # Libraries
    from matplotlib.patches import Polygon

    # Construct variables
    means = np.concatenate([
        gmm.means_ for gmm in gmms
    ])
    covs = np.concatenate([
        gmm.covariances_ for gmm in gmms
    ])

    # Create ellipses (shapely)
    ells_shp = [
        create_ellipse_shapely(m, c)
        for m, c in zip(means, covs)]

    # Create ellipses (matplotlib)
    ells_mpl = [
        create_ellipse_mpl(m, c)
        for m, c in zip(means, covs)]

    # Create figure
    fig, ax = plt.subplots()
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c=y)

    # Display ellipses (shapely)
    for e in ells_shp:
        verts1 = np.array(e.exterior.coords.xy)
        patch1 = Polygon(verts1.T, color='blue', alpha=0.25)
        ax.add_patch(patch1)

    # Display ellipses (matplotlib)
    for e in ells_mpl:
        ax.add_artist(e)

    # Show
    plt.show()