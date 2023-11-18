import numpy as np
from cvxopt import matrix
from cvxopt import sparse
from cvxopt.solvers import qp


def linear_kernel(X, Y=None):
    return np.dot(X, X.T) if Y is None else np.dot(X, Y.T)


class ISVDD:
    '''
    This class implements SVDD+ algorithm,
    imitates sklearn interface
    Right now it supports only linear kernel        
    Parameters:
    - nu: float
        The upper bound on the fraction of margin errors and the fraction
        of support vectors.
    - features_kernel: function, optional (default=linear_kernel)
        The kernel function for the feature space.
    - privileged_kernel: function, optional (default=linear_kernel)
        The kernel function for the privileged space.
    - privileged_regularization: float, optional (default=0.1)
        Regularization parameter for the privileged space.
    - tol: float, optional (default=0.001)
        Tolerance for convergence of the optimization problem.
    - max_iter: int, optional (default=100)
         Maximum number of iterations for the optimization problem.
    - silent: bool, optional (default=True)
        If True, suppress optimization progress output.
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 tol=0.001, max_iter=100, silent=True):
        '''
        This class implements SVDD+ algorithm,
        imitates sklearn interface
        Right now it supports only linear kernel        
        Parameters:
        - nu: float
            The upper bound on the fraction of margin errors and the fraction
            of support vectors.
        - features_kernel: function, optional (default=linear_kernel)
            The kernel function for the feature space.
        - privileged_kernel: function, optional (default=linear_kernel)
            The kernel function for the privileged space.
        - privileged_regularization: float, optional (default=0.1)
            Regularization parameter for the privileged space.
        - tol: float, optional (default=0.001)
            Tolerance for convergence of the optimization problem.
        - max_iter: int, optional (default=100)
            Maximum number of iterations for the optimization problem.
        - silent: bool, optional (default=True)
            If True, suppress optimization progress output.
        '''
        # Setting initial parameters
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.privileged_regularization = privileged_regularization
        self.tol = tol
        self.silent = silent
        # Initializing with None some futer parameters
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None

    def _prepare_problem(self, X, Z):
        """
        Prepare matrices for quadratic optimization problem

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
          Feature matrix.
        - Z: numpy array, shape (n_samples, n_features)
          Privileged information matrix.

        Returns:
        - dict
          Dictionary containing matrices for the optimization problem.
        """
        gamma = self.privileged_regularization
        C = 1.0 / len(X) / self.nu
        size = X.shape[0]
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        zeros_matrix = np.zeros_like(kernel_x)
        P = 2 * np.bmat([[kernel_x, zeros_matrix],
                         [zeros_matrix, 0.5*gamma*kernel_z]])
        P = matrix(P)
        q = matrix(list(np.diag(kernel_x)) + [0] * size)
        A = matrix([[1.]*size + [0.]*size, [1.] * size*2]).T
        b = matrix([1., 1.])
        G = np.bmat([[-np.eye(size), zeros_matrix],
                     [-np.eye(size), np.eye(size)],
                     [np.eye(size), -np.eye(size)]])
        G = matrix(G)
        G = sparse(G)
        h = matrix([0]*size*2 + [C]*size)
        optimization_problem = {'P': P, 'q': q, 'G': G,
                                'h': h, 'A': A, 'b': b}
        return optimization_problem

    # Helper function for prediction
    def _scalar_product_with_center(self, X):
        return np.dot(self.features_kernel(X, self.support_vectors),
                      self.dual_alpha)

    def _calculate_threshold(self):
        """
        Calculates critical value of the decision function
        """
        kernel_support = self.features_kernel(self.support_vectors)
        self.centre_norm = np.dot(self.dual_alpha,
                                  np.dot(kernel_support,
                                         self.dual_alpha))
        # Select the first support vector since distance
        # between it center equal to R
        single_support_vector = self.support_vectors[0, :]
        first_support_vector = single_support_vector.reshape(1, -1)
        support_vector_norm = self.features_kernel(first_support_vector)
        dot_product_with_centre = 2 * self._scalar_product_with_center(
                                          single_support_vector[np.newaxis, :])
        self.radius = (support_vector_norm +
                       self.centre_norm -
                       dot_product_with_centre)
        self.threshold = self.centre_norm - self.radius

    def fit(self, X, Z):
        '''
        Method takes matrix with feature values
        and information from privileged feature
        space, solves optimization problem and
        calculates center and radius of the sphere.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
          Feature matrix.
        - Z: numpy array, shape (n_samples, n_features)
          Privileged information matrix.
        '''
        problem = self._prepare_problem(X, Z)
        options = {}
        options['show_progress'] = not self.silent
        options['maxiters'] = self.max_iter
        options['abstol'] = self.tol
        problem['options'] = options
        solver = qp(**problem)
        if solver['status'] != 'optimal':
            raise ValueError("Failed Optimization")
        self.dual_solution = np.array(solver['x']).reshape(2*len(X),)
        self.support_indices = np.where(self.dual_solution[:len(X)] > 0)[0]
        self.support_vectors = X[self.support_indices, :]
        self.dual_alpha = self.dual_solution[self.support_indices]
        self._calculate_threshold()
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
          Feature matrix.

        Returns:
        - numpy array, shape (n_samples,)
          Anomaly scores for the input points.
        """
        test_norm = np.diag(self.features_kernel(X))
        scalar_product = self._scalar_product_with_center(X)
        return test_norm.ravel()[0, :] + self.threshold - 2*scalar_product
