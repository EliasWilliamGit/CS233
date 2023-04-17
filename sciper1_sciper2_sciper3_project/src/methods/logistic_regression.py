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
        pred_labels = self.logistic_regression_train_multi(self, training_data, training_labels)
        
        return pred_labels
        

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
        pred_labels = self.logistic_regression_predict_multi(self, test_data)
        
        return pred_labels
    
    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
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
    

    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
    
        y = self.f_softmax(data, W)

        return data.T @ (y - labels)
    
    def logistic_regression_predict_multi(self, data):
        """
        Prediction the label of data for multi-class logistic regression.
    
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
    
        y = self.f_softmax(data)
        return np.argmax(y, axis = 1)
    
    def logistic_regression_train_multi(self, data, labels):
        """
        Training function for multi class logistic regression.
    
        Args:
            data (array): Dataset of shape (N, D).
            labels (array): Labels of shape (N, C)
            max_iters (int): Maximum number of iterations. Default: 10
            lr (int): The learning rate of  the gradient step. Default: 0.001
            
        Returns:
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        D = data.shape[1]  # number of features
        C = labels.shape[1]  # number of classes
        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            
            gradient = self.gradient_logistic_multi(data, labels, weights)
            weights = weights - self.lr * gradient
            
            predictions = self.logistic_regression_predict_multi(data, self.weights)
            if self.accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                break
            
        self.weights = weights
        return predictions
            
    