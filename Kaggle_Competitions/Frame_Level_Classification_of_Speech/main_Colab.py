from __future__ import print_function
import argparse
import torch
import torch.nn
import torch.optim as optim
import time
import numpy as np
import itertools

#Load the datasets
features_dev = np.load('dev.npy', allow_pickle = True)
labels_dev = np.load('dev_labels.npy', allow_pickle = True)
features_train = np.load('train.npy', allow_pickle = True)
labels_train = np.load('train_labels.npy', allow_pickle = True)
features_test = np.load('test.npy', allow_pickle = True)

'''Class used to load the features and labels for both the training and validation(dev) datasets'''
class Data(torch.utils.data.Dataset):
    def __init__(self, feature_set, label_set, context_length=12):
        self.context_length = context_length  #essentiall, number of features extracted to examine at once
        self.features = feature_set #features of dataset
        self.features_small = self.features  #was used to trim data set by slicing portions of entire set (would slice portion of data)
        self.labels = label_set #labels of dataset
        self.map = []   #initialize index mapping

        for i, utterance in enumerate(self.features_small):  #iterate over the utterances
            for j in range(utterance.shape[0]): #iterate over rows
                self.map.append((i, j)) #append an utterance ID and row ID for extraction in _getitem_

    def __getitem__(self, index_val):
        i, j = self.map[index_val]    #locate utterance and row
        context_range = range(j - self.context_length, j + self.context_length + 1) #context range will give indices of the rows to extract. Neg. values will extract mirror over edge
        elements = self.features_small[i].take(context_range, mode='clip', axis=0).flatten() #extract rows and flatten data
        feature_of_utter = torch.from_numpy(elements).float() #turn data into pytorch tensor
        label = self.labels[i][j]   #acquire labels from index
        return feature_of_utter, label

    def __len__(self):
        return len(self.map) #define length
'''This data class is used specifically for the test data. It is indexed the exact same but only contains the features as we are predicting the labels'''
class Test_Data(torch.utils.data.Dataset):
    def __init__(self, feature_set, context_length=12):
        self.context_length = context_length
        self.features = feature_set
        self.features_small = self.features
        self.map = []

        for i, utterance in enumerate(self.features_small):
            for j in range(utterance.shape[0]):
                self.map.append((i, j))

    def __getitem__(self, index_val):
        i, j = self.map[index_val]
        context_range = range(j - self.context_length, j + self.context_length + 1)
        elements = self.features_small[i].take(context_range, mode='clip', axis=0).flatten()
        feature_of_utter = torch.from_numpy(elements).float()
        return feature_of_utter

    def __len__(self):
        return len(self.map)

'''This function defines the feed forward model created. My model that gave me the best results from testing is shown below.
I ended up going with 7 hidden layers and using BN before the ReLU activation. Hidden sizes were based around what the baseline 
model on piazza stated then tweaked. Different combinations were tested.
'''
def feed_forward_model(context_length):
    input_size = (2 * context_length + 1) * 40 #based the input size of the context length (ends up being 1000 with a context length of 12)
    hidden_sizes = [input_size * n for n in range(1,3)] #used to get hidden layer sizes
    output_size = 138  #num_classes
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_sizes[1]),
        torch.nn.BatchNorm1d(hidden_sizes[1]),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[1], hidden_sizes[1]),
        torch.nn.BatchNorm1d(hidden_sizes[1]),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[1], hidden_sizes[1]),
        torch.nn.BatchNorm1d(hidden_sizes[1]),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[1], hidden_sizes[0]),
        torch.nn.BatchNorm1d(hidden_sizes[0]),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[0], hidden_sizes[0]),
        torch.nn.BatchNorm1d(hidden_sizes[0]),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[0], hidden_sizes[0]//2),
        torch.nn.BatchNorm1d(hidden_sizes[0]//2),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[0] // 2, hidden_sizes[0] // 2),
        torch.nn.BatchNorm1d(hidden_sizes[0] // 2),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_sizes[0]//2, output_size),
    )
'''Function used to train the model. Returns the running loss for calculation purposes in test function. This function was 
structured based on Recitation 1 code.'''
def train(args, model, training_data_loader, criterion, optimizer, epoch, device):
    model.train()  #initialize training of model
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(training_data_loader):  #loop over the training data
        optimizer.zero_grad()       #clear gradients
        data = data.to(device)      #make data run on device (GPU or CPU)
        target = target.to(device)  #make target data(labels) run on device
        outputs = model(data)       #forward pass through model to get predictions from our data(features)
        loss = criterion(outputs, target) #get loss from forward pass and the target values
        running_loss += loss.item()     #sum the loss
        loss.backward()             #backward pass through to calculate weights and bias updates needed
        optimizer.step()            #step and update weights to linear and BN layers

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} Batch Number: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), loss.item()))

        #The information above was used as a percentage calculator to see the script was running through with some information as well
    end_time = time.time()
    running_loss /= len(training_data_loader)   #calculate running loss
    print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
    return running_loss
''' Test function gives us the relationship between our target output from our model and the given labels from the dataset. Modelled after 
recitation 1 code'''
def test(model, dev_data_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        for batch_idx, (data, target) in enumerate(dev_data_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)  #calculate number of predicitons made
            correct_predictions += (predicted == target).sum().item() #sum number of correct predictions
            loss = criterion(outputs, target).detach()  #calculate loss from torch.nn.CrossEntropy
            running_loss += loss.item()
        running_loss /= len(dev_data_loader)
        acc = (correct_predictions / total_predictions) * 100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
'''This function is our label predictor once the model has been trained. Similar to function above but removes portions with labels'''
def test_predictions(model, test_data_loader, device):  
    model.eval()   
    predictions = []
    predictions_final = []
    for data in test_data_loader: #iterate through test data
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1) #acquire predicted values
        predictions.append(predicted.cpu().numpy().flatten()) #append predictions after turning to numpy and flattened
    predictions_final = np.asarray(np.asmatrix(list(itertools.chain.from_iterable(predictions))).T) #used to give us a column of our predictions
    columns = np.asarray(np.asmatrix(range(len(predictions_final))).T)  #column indices
    np.savetxt('Kaggle_Submission6HW1.csv', np.c_[columns, predictions_final], delimiter=',', fmt='%d', header="id,label", comments='') #save a csv file

def main():
    #Training settings for the model(Simple place to modify some params)
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training if true')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='number of batches to wait before logging')
    parser.add_argument('--context_length', type=int, default=12, metavar='N',
                        help='Context Size')
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()   #check if GPU is available
    torch.manual_seed(args.seed)        #set seed manually
    device = torch.device("cuda:0" if cuda else "cpu") #gives whether the device is CPU or GPU

    if cuda:
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    '''Data loaders below call in the data set, and give other arguments like batch size, shuffle, num_workers, etc.'''
    training_data_loader = torch.utils.data.DataLoader(Data(features_train,labels_train, args.context_length), batch_size=args.train_batch_size, shuffle=True, **kwargs)
    dev_data_loader = torch.utils.data.DataLoader(Data(features_dev,labels_dev, args.context_length), batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(Test_Data(features_test, args.context_length), batch_size=args.train_batch_size, shuffle=False, **kwargs)

    model = feed_forward_model(args.context_length).to(device) #define model
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #set optimizer and learning rate
    criterion = torch.nn.CrossEntropyLoss() #define criterion for loss
    n_epochs = 15           #set num epochs
    Train_loss = []
    Test_loss = []
    Test_acc = []
    current_accuracy = 0.0
    for i in range(n_epochs):
        epoch = i + 1 #value used for giving extra informaition in training function to help ensure running correctly.
        train_loss = train(args, model, training_data_loader, criterion, optimizer, epoch, device) #train model
        test_loss, test_acc = test(model, dev_data_loader, criterion, device) #test on validation data
        if test_acc < current_accuracy: #if dips below the previous test accuracy, break.
            break
        else:
            current_accuracy = test_acc
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        print('=' * 20)
    test_predictions(model, test_data_loader, device) #acquire predictions from trained model
    print('=' * 20)
if __name__ == '__main__':
    main()