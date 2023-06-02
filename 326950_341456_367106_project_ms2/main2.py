import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, label_to_onehot
import matplotlib.pyplot as plt


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our images
    xtrain, xtest, ytrain, ytest = load_data(args.data)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    np.random.seed(0)
    shuffled_id = np.random.permutation(xtrain.shape[0])
    xtrain = xtrain[shuffled_id]
    ytrain = ytrain[shuffled_id]

    # Make a validation set
    if not args.test:
        # Split the shuffled training data into training and validation sets
        num_train = int(0.8 * xtrain.shape[0])  # 80% of data for training
        xtrain, xval = xtrain[:num_train], xtrain[num_train:]
        ytrain, yval = ytrain[:num_train], ytrain[num_train:]


        # 80 % of the samples dedicated for training, the rest for validation
        train_split = np.floor(4/5 * xtrain.shape[0]).astype(int)
        xval = xtrain[train_split:]
        xtrain = xtrain[:train_split]

        yval = ytrain[train_split:]
        ytrain = ytrain[:train_split]

    # Normalization of all the data, using only the mean/std from the training set
    mean = np.mean(xtrain, axis = 0)
    std = np.std(xtrain, axis = 0)

    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean , std)


    if not args.test:
        xval = normalize_fn(xval, mean, std)

    """
    # Append a bias term to the data
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)
    if not args.test:
        xval = append_bias_term(xval)
    """

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        # Flattening the data before running PCA
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        if not args.test:
            xval = xval.reshape(xval.shape[0], -1)
        d=args.pca_d
        pca_obj = PCA(d=args.pca_d)
        ### Use the PCA object to reduce the dimensionality of the data
        var = pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        if not args.test:
            xval = pca_obj.reduce_dimension(xval)
        print(f"Explained Variance for d={d} dimensions (D=1024) : {var:.3f}%")

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)

        if args.nn_type == "mlp":

            # Flattening the data before running 
            xtrain = xtrain.reshape(xtrain.shape[0], -1)
            xtest = xtest.reshape(xtest.shape[0], -1)
            if not args.test:
                xval = xval.reshape(xval.shape[0], -1)
            input_size = xtrain.shape[1]
            model = MLP(input_size = input_size , n_classes= n_classes)   

        elif args.nn_type == "cnn":
            if args.use_pca:
                print("Cannot use PCA with CNN.")
                return
            
            input_channels = 1

            # Add channel dimension to image (For pythorch to work)
            xtrain = xtrain[:,np.newaxis,:,:]
            xtest = xtest[:,np.newaxis,:,:]
            if not args.test:
                xval = xval[:,np.newaxis,:,:]
            
            model = CNN(input_channels=input_channels, n_classes=n_classes)
        
        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size,
                             weight_decay=args.weight_decay, use_lr_scheduler=args.use_lr_scheduler)
    
    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)
    

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data TODO: add validation data here for cross val accuracy, maybe bool
    if args.test:
        preds_train = method_obj.fit(xtrain, ytrain)
    else:
        preds_train = method_obj.fit_val(xtrain, ytrain, xval, yval)
        
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
    if args.test:
        print("ERROR: Cannot do visualize and test at the same time")
        return
    
    if args.pca_d_visualize :
        # Flattening the data before running PCA
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        if not args.test:
            xval = xval.reshape(xval.shape[0], -1)
        
        # Plot of the total variance explained by the first d principal components

        # Creating the var array with the explained variance for each dimension (from 2 to d)
        i=0
        var = np.empty((args.pca_d, 2))
        for a in range(2, args.pca_d):
            pca_obj = PCA(a)
            var[i,0] = i
            var[i,1] = pca_obj.find_principal_components(xtrain)
            i += 1

        # Plot
        x = var[:,0] 
        y = var[:,1] 

        plt.bar(x, y)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance of the d Principal Components')
        plt.show()
        return
    
    if args.method == "nn":
        if args.nn_type == "mlp":
            return
        elif args.nn_type == "cnn":
            val_acc_list, train_acc_list, loss_list = method_obj.get_training_info()
            epoch_list = list(range(1, len(val_acc_list) + 1))

            fig, ax = plt.subplots(2, 1, figsize=(6,6))
            ax[0].plot(epoch_list, train_acc_list, color="r", label="Training accuracy")
            ax[0].plot(epoch_list, val_acc_list, color="b", label="Validation accuracy")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Accuracy")
            ax[0].legend()

            ax[1].plot(epoch_list, loss_list, color="b")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss value")

            plt.show()
            return
        

    




if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / pca (MS2) / nn (MS2)")
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
    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--pca_d_visualize', action="store_true", help="to visualize which d is the best for the train set")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--use_lr_scheduler', action="store_true", help="Reduce learning rate with training")
    parser.add_argument('--weight_decay', type=float, default=0, help="Value for the regularization term during training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
