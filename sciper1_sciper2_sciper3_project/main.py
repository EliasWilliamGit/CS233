import argparse

import numpy as np 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        # 80 % of the samples dedicated for training, the rest for validation
        train_split = np.floor(4/5 * xtrain.shape[0]).astype(int)
        xval = xtrain[train_split:]
        xtrain = xtrain[:train_split]

        yval = ytrain[train_split:]
        ytrain = ytrain[:train_split]
    
    ### WRITE YOUR CODE HERE to do any other data processing

    # Normalization of all the data, using only the mean/std from the training set
    mean = np.mean(xtrain, axis = 0)
    std = np.std(xtrain, axis = 0)

    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean , std)

    if not args.test:
        xval = normalize_fn(xval, mean, std)

    # Append a bias term to the data
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)
    if not args.test:
        xval = append_bias_term(xval)
    

    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":  ### WRITE YOUR CODE HERE
        method_obj = KMeans(K = args.K, max_iters=args.max_iters)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters=args.max_iters)

    elif args.method == "svm":
        method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0 )   

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
    
    # Predict on unseen data
    if args.test:
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    if not args.test:
        preds = method_obj.predict(xval)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, yval)
        macrof1 = macrof1_fn(preds, yval)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if not args.visualize:
        return
    
    if args.method == "kmeans":
        # Calculate accuracy and f1 score based on the vakue of K

        K_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        acc_train = []
        acc_val = []
        f1_train = []
        f1_val = []

        for K in K_values:
            print(f'Calculating for K = {K}...')
            method_obj = KMeans(K = K)
            preds_train = method_obj.fit(xtrain, ytrain)
            preds_val = method_obj.predict(xval)

            # Save test accuracy and f1 score
            acc_train.append(accuracy_fn(preds_train, ytrain))
            f1_train.append(macrof1_fn(preds_train, ytrain))

            # Save validation accuracy and f1 score
            acc_val.append(accuracy_fn(preds_val, yval))
            f1_val.append(macrof1_fn(preds_val, yval))

        # Visualize the accuracies and f1 scores dependent on K
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(K_values, acc_train, color="r", label="Test accuracy")
        ax[0].plot(K_values, acc_val, color="b", label="Validation accuracy")
        ax[0].set_xlabel("K")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(K_values, f1_train, color="r", label="Test f1")
        ax[1].plot(K_values, f1_val, color="b", label="Validation f1")
        ax[1].set_xlabel("K")
        ax[1].set_ylabel("f1 score")
        ax[1].legend()

        plt.show()

    if args.method == "logistic_regression":
        return

    if args.method == "svm":
        # what should i iterate on to find the best fitting method ? 
        C = []
        kernel = []
        gamma = 1
        degree = 1
        coef0 = 0
        





if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")
    parser.add_argument('--visualize', action="store_true", help="Visualize the method with different values for its hyperparameter(s)")

    # Feel free to add more arguments here if you need!

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
