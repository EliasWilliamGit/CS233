import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0).reshape(1,-1)

        # Center the data with the mean
        training_data_tilde = training_data - self.mean

        # Create the covariance matrix
        C = np.cov(training_data_tilde, rowvar=False)

        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(C)

        # Choose the top d eigenvalues and corresponding eigenvectors. 
        sorted_id = np.argsort(eigvals)[::-1]
        sorted_d_id = sorted_id[:self.d] #The resulting sorted_n_indices array contains the indices of the d largest eigenvalues.

        sorted_d_eigvals = eigvals[sorted_d_id]
        self.W = eigvecs[:, sorted_d_id]

        # Compute the explained variance
        exvar = 100 * np.sum(sorted_d_eigvals)/np.sum(eigvals)
        # calculates the explained variance as a percentage. The numerator np.sum(eigvals) computes the sum of the selected 
        # eigenvalues (i.e., the top d eigenvalues), while the denominator np.sum(eigvals_full) computes the sum of all the eigenvalues.

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        return data_reduced
        

