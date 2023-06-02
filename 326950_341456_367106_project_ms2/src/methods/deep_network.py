import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.utils import *
import timeit
## MS2
    
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
            
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        # Check if the 
        self.fc1 = nn.Linear(in_features = input_size, out_features = 400) # First fully connected layer
        self.fc2 = nn.Linear(in_features = 400, out_features = 100) # Second fully connected layer 
        self.fc3 = nn.Linear(in_features = 100, out_features = n_classes) # Output layer

        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        x = F.relu(self.fc1(x)) # Apply ReLU activation to the first layer output
        x = F.relu(self.fc2(x)) # Apply ReLU activation to the second layer output
        preds = self.fc3(x) # Output layer (no activation function)
        return preds
    
    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)



class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        # Definition of the layers
        self.conv2d1 = nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 3, padding = 1)
        self.conv2d2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=n_classes)
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        # Forward pass through the network
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(kernel_size=2, input=x)
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(kernel_size=2, input=x)
        x = x.flatten(-3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        preds = self.fc3(x)

        return preds
    
    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, weight_decay = 0, use_lr_scheduler = False):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        if use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters= self.epochs - 1)

        self.loss_list=[]
        self.train_acc_list=[]
        self.val_acc_list=[]

    def get_training_info(self):
        
        return self.val_acc_list, self.train_acc_list, self.loss_list

    def train_all(self, dataloader_train, dataloader_val=False):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        # Put model in training mode
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train, ep)

            ### WRITE YOUR CODE HERE if you want to do add else at each epoch

            # Reduce learning rate if we use learning rate scheduler
            if self.use_lr_scheduler:
                self.lr_scheduler.step()

            if dataloader_val:
                # Validate model at each epoch
                self.val_one_epoch(dataloader_val, ep)
            
    def val_one_epoch(self, dataloader,ep):
        """
        Validate the model for ONE epoch.

        Should loop over the batches in the dataloader.

        Arguments:
            dataloader (DataLoader): dataloader for validation data
            ep (int): current epoch for prints
        """
        self.model.eval()

        with torch.no_grad():
            # Loop over all batches in the val/test set
            acc_run = 0

            for it, batch in enumerate(dataloader):
                x,y = batch
                if torch.cuda.is_available():
                    y = y.type(torch.cuda.LongTensor)
                else:
                    y = y.type(torch.LongTensor)

                # Predict on the batch
                logits = self.model.predict(x)

                curr_bs = x.shape[0]
                acc_run += accuracy_fn(onehot_to_label(logits.detach().cpu().numpy()), y.detach().cpu().numpy()) * curr_bs
                
        acc = acc_run / len(dataloader.dataset)

        # Add them to list for validation later
        self.val_acc_list.append(acc)

        # Print performance
        print(f'Ep {ep+ 1}/{self.epochs}: accuracy val: {acc}')

        self.model.train()


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
            ep (int): current epoch for prints
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        acc_run = 0
        loss_run = 0
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch
            # y = torch.tensor(y, dtype=torch.int64)
            if torch.cuda.is_available():
                y = y.type(torch.cuda.LongTensor)
            else:
                y = y.type(torch.LongTensor)

            # 5.2 Run forward pass.
            logits = self.model.forward(x)
            
            # 5.3 Compute loss (using 'criterion').

            loss = self.criterion(logits, y)
            
            # 5.4 Run backward pass.
            loss.backward()  ### WRITE YOUR CODE HERE^
            
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()  ### WRITE YOUR CODE HERE
            
            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()  ### WRITE YOUR CODE HERE^

            curr_bs = x.shape[0]
            acc_run += accuracy_fn(onehot_to_label(logits.detach().cpu().numpy()), y.detach().cpu().numpy()) * curr_bs
            loss_run += loss.item() * curr_bs

        acc = acc_run / len(dataloader.dataset)
        loss_mean = loss_run / len(dataloader.dataset)

        # Add them to list for validation later
        self.train_acc_list.append(acc)
        self.loss_list.append(loss_mean)

        # Print performance
        print(f'Ep {ep+ 1}/{self.epochs}: loss train: {loss_mean}, accuracy train: {acc}')


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        # Evaluation mode
        self.model.eval()
        N = len(dataloader.dataset)
        pred_labels = torch.zeros(N)

        with torch.no_grad():
            # Loop over all batches in the val/test set
            for it, batch in enumerate(dataloader):
                x = batch[0]

                # Predict on the batch
                pred_label = self.model.predict(x)
                # Convert to label
                pred_label = onehot_to_label(pred_label.detach().cpu().numpy())
                pred_label = torch.tensor(pred_label)
                
                pred_labels[it*self.batch_size:(it + 1)*self.batch_size] = pred_label
        
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch

        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        start = timeit.default_timer()
        self.train_all(train_dataloader)
        stop = timeit.default_timer()

        print(f"Time elapsed training: {stop - start} s")

        return self.predict(training_data)
    
    def fit_val(self, training_data, training_labels, validation_data, validation_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
            validation_data (array): validation data of shape (N,D)
            validation_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.from_numpy(validation_data).float(), 
                                      torch.from_numpy(validation_labels))
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        
        start = timeit.default_timer()
        self.train_all(train_dataloader, val_dataloader)
        stop = timeit.default_timer()

        print(f"Time elapsed training, using the validation step: {stop - start} s")

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        start = timeit.default_timer()
        pred_labels = self.predict_torch(test_dataloader)
        stop = timeit.default_timer()

        print(f"Time elapsed inference: {stop - start} s")

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()