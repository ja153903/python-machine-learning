from scipy.spatial.distance import pdist, squareform
from scipy import exp 
from scipy.linalg import eigh
import numpy as np 
from sklearn.datasets import make_moons 
import matplotlib.pyplot as plt 

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation 

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel
    
    n_components: int
        Number of principal components to return
    
    Return
    -------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        Projected dataset
    """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N 
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) 

    # Obtain eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    return X_pc

from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
                        color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
                        color='blue', marker='o', alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# if __name__ == '__main__':
    # X, y = make_moons(n_samples=100, random_state=123)
    # plt.scatter(X[y==0, 0], X[y==0, 1],
    #             color='red', marker='^', alpha=0.5)
    # plt.scatter(X[y==1, 0], X[y==1, 1],
    #             color='blue', marker='o', alpha=0.5)
    # plt.show()