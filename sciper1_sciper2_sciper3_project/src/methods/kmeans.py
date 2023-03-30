import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters

        # Create this to be used in predict. Dont know if this is a good init. Doesnt know sec dim.
        self._centers = np.zeros((K,1))
        self._cluster_center_labels = np.zeros(K)

    def k_means(self, data, max_iter):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        # Initialize the centers
        centers = self.__init_centers(data)
        
        D = data.shape[1]

        # Loop over the iterations
        for i in range(max_iter):
            
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}...")
                
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration
            
            # Distances from data to the centers
            distances = self.__compute_distance(data, centers)
            
            # Find the closest center to each sample
            cluster_assignments = self.__find_closest_cluster(distances)
            
            # Update the centers
            centers = self.__compute_centers(data, cluster_assignments)
            
            # End of the algorithm if the centers have not moved
            if np.all(centers == old_centers):
                print(f"K-Means has converged after {i+1} iterations!")
                break
        
        # Compute the final cluster assignments
        
        # Distances from data to the centers
        distances = self.__compute_distance(data,centers)
        
        cluster_assignments = self.__find_closest_cluster(distances)

        return centers, cluster_assignments
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        self._centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        print(f'Cluster assignments: {cluster_assignments}')
        self._cluster_center_labels = self.__assign_labels_to_centers(self._centers, cluster_assignments, training_labels)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        # Calculate the distance for the samples to the cluster centers
        distances = self.__compute_distance(test_data,self._centers)
    
        # Pick the closest cluster for the samples
        cluster_assignments = self.__find_closest_cluster(distances)
        
        N = test_data.shape[0]
        K = self._centers.shape[0]
        
        pred_labels = np.zeros(N)
        
        for cluster_index in range(K):

            # Get mask of the samples belonging to cluster cluster_index
            indicies = cluster_assignments == cluster_index

            # Get the real label for this cluster
            predicted_label = self._cluster_center_labels[cluster_index]
           
            # Save the real label for all samples belonging to this cluster
            pred_labels[indicies] = predicted_label

        return pred_labels
    
    def __init_centers(self, training_data):
        """
        Inititalize the centers randomly as some training samples.

        Arguments: 
        data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
    Returns:
        centers: array of shape (KxD) of initial cluster centers
        """
        # Create K random indexes in the range of the data
        N = training_data.shape[0]
        random_idxs = np.random.permutation(range(N))[:self.K]

        # Pick K random samples as initial centers
        centers = training_data[random_idxs]

        return centers
    def __compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        N = data.shape[0]
        
        distances = np.zeros((N, self.K))
        
        for k in range(self.K):
            # Compute the euclidean distance for each data to each center
            center = centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))

        return distances
    
    def __find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.
        
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        cluster_assignments = np.argmin(distances, axis=1)

        return cluster_assignments
    
    def __compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """

        D = data.shape[1]
        centers = np.zeros((self.K,D))
        
        # Loop through the centers and calculate the mean for the cluster
        for cluster_center in range(self.K):
            # Get number of points assigned to this cluster
            Nk = np.sum(cluster_assignments == cluster_center)
            
            # Get the indicies of the points assigned to the cluster
            binary_indicis = cluster_assignments == cluster_center
            
            # Get the points assigned to the cluster
            points_assigned = data[binary_indicis,:]
            
            # Calculate the new center based on the assigned points
            centers[cluster_center] = 1/Nk * np.sum(points_assigned, axis = 0)
        
        return centers
    
    def __assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        K = centers.shape[0]
        cluster_center_label = np.zeros(K)
        
        
        for center in range(K):
            
            # Get mask for samples in this cluster
            indicis = cluster_assignments == center
            # Get the true labels for the samples
            true_labels_for_points = true_labels[indicis]

            # Count the labels
            occourencies = np.bincount(true_labels_for_points)

            # Pick the most frequent label
            predicted_label = np.argmax(occourencies)
            
            cluster_center_label[center] = predicted_label

        return cluster_center_label
    
