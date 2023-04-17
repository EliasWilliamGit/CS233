import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """
    
    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

        
    def f_softmax(data):
        """""
        Softmax function for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)

        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
      
        num = np.exp(data @ W)
        denum = np.sum(num, axis=1)
        N = data.shape[0]
        denum = denum.reshape((N,1))
    
        return num/denum
        
    def loss_logistic_multi(self, data, labels):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            
        Returns:
            float: Loss value 
        """
        
        ln_y = np.log(self.f_softmax(data, self.w))

        return - np.sum(labels * ln_y)


    def gradient_logistic_multi(self, data, labels):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
    
        y = self.f_softmax(data, self.weights)

        return data.T @ (y - labels)


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        D = training_data.shape[1]  # number of features
        C = training_labels.shape[1]  # number of classes

        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            
            gradient = self.gradient_logistic_multi(training_data, training_labels, weights)
            weights = weights - self.lr * gradient

        self.weights = weights
             
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
    
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        y = self.f_softmax(test_data, self.weights)
        pred_labels = np.argmax(y, axis = 1)

        return pred_labels