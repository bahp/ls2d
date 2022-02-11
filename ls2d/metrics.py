# Generic
import numpy as np

# Scipy
from scipy import linalg


def gmm_intersection_matrix(y_pred, y, **params):
    """Compute the GMM intersection area matrix.

    Parameters
    ----------
    y_pred: np.ndarray
        The projections in the embedded space.
    y: np.ndarray
        The labels to compute the GMMs.
    **params:
        Parameters to pass to the XX function

    Returns
    -------
    """
    # Libraries
    from sklearn.mixture import BayesianGaussianMixture

    # Fit GMMs.
    gmms = [
        BayesianGaussianMixture(n_components=1,
            covariance_type='full', **params) \
                .fit(y_pred[y == t])
                    for t in np.unique(y)
    ]

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

    # Compute matrix
    n = len(ells_shp)
    matrix = np.zeros((n, n))
    for i, p1 in enumerate(ells_shp):
        for j, p2 in enumerate(ells_shp):
            matrix[i, j] = p1.intersection(p2).area

    # Normalize (divide by matrix size?)
    # Return sum of all

    # Return
    return matrix, np.unique(y), gmms

def gmm_ratio_score(y_pred, y, method='sum', **params):
    """Compute the GMM ratio score.

    Parameters
    ----------
    y_pred: np.ndarray
        The projections in the embedded space.
    y: np.ndarray
        The labels to compute the GMMs.
    **params:
        Parameters to pass to the XX function

    Returns
    -------
    """
    # Create matrix
    gmm_matrix, label, gmms = \
        gmm_intersection_matrix(y_pred, y, **params)
    # Intersection ratios
    gmm_ratios = gmm_matrix / np.diag(gmm_matrix)
    # Intersection ratios sum
    idxs = np.where(~np.eye(gmm_ratios.shape[0], dtype=bool))
    # Return
    if method=='sum':
        return np.sum(gmm_ratios[idxs])
    elif method=='avg':
        return np.mean(gmm_ratios[idxs])
    else:
        return -1



def ellipse_params(means, covariances):
    """Compute ellipse parameters from mean and covariances.

    Parameters
    ----------
    means: np.ndarray
        The means.
    covariances: np.ndarray
        The covariances.

    Returns
    -------
    center, variance, angle
    """
    # Compute
    v, w = linalg.eigh(covariances)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi
    # Returns
    return means, v, angle


def create_ellipse_shapely(means, covariances, factor=1):
    """Create an ellipse (Polygon object)

    Parameters
    ----------
    means: np.ndarray
        The means.
    covariances: np.ndarray
        The covariances.

    Returns
    -------
    """
    # Libraries
    from shapely.geometry.point import Point
    from shapely import affinity

    # Compute parameters
    center, v, angle = ellipse_params(means, covariances)

    # Create Polygon
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, v[0]/factor, v[1]/factor)
    ell = affinity.rotate(ell, angle)

    # Return
    return ell


def create_ellipse_mpl(means, covariances):
    """Create an ellipse (Ellipse object).

    Parameters
    ----------
    means: np.ndarray
        The means.
    covariances: np.ndarray
        The covariances.

    Returns
    -------
    """
    # Libraries
    from matplotlib.patches import Ellipse

    # Compute parameters
    center, v, angle = ellipse_params(means, covariances)

    # Create polygon
    ell = Ellipse(means, v[0], v[1], 180.+angle,
        fill=False, lw=2, ls='--')

    # Return
    return ell







if __name__ == '__main__':


    # ------------------
    # Libraries
    # ------------------
    # Generic libraries
    import numpy as np

    # Basic Functions in Sklearn
    from sklearn import datasets
    from sklearn import metrics
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.decomposition import PCA

    # Matplotlib
    import matplotlib.pyplot as plt

    # ------------------
    # Data
    # ------------------
    # Create data
    X, y = iris = datasets.load_iris(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Transformer
    pca = PCA(n_components=2).fit(X)

    # Create variables
    y_pred = pca.transform(X)


    # ------------------
    # Metrics
    # ------------------
    # Intersection matrix
    gmm_matrix, label, gmms = gmm_intersection_matrix(y_pred, y)
    # Intersection ratios
    gmm_ratios = gmm_matrix / np.diag(gmm_matrix)
    # Intersection normalization
    ratio_sum = np.sum(gmm_ratios) - sum(gmm_ratios[np.diag_indices(3)])
    #ratio_avg = ratio_sum /

    # Show
    print("\nAreas:")
    print(gmm_matrix)
    print("\nRatios:")
    print(gmm_ratios)
    print("\nRatio (sum):")
    print(ratio_sum)

    # Compute silhouette
    silhouette = metrics.silhouette_score(y_pred, y, metric="sqeuclidean")
    homo = metrics.homogeneity_score(y, y)
    comp = metrics.completeness_score(y, y)
    vmeasure = metrics.v_measure_score(y, y)
    adrs = metrics.adjusted_rand_score(y, y)
    admis = metrics.adjusted_mutual_info_score(y, y)

    # Show
    print("\n\n")
    print("Shilhouette:  %s" % silhouette)
    print("Homogenity:   %s" % homo)
    print("Completeness: %s" % comp)
    print("V-Measure:             %s" % vmeasure)
    print("Adj Rand Score:        %s" % adrs)
    print("Adj Mutual Info Score: %s" % admis)

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