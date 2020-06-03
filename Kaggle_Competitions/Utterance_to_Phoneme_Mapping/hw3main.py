from __future__ import print_function
import torch
import torch.nn
import torch.optim as optim
import time
import numpy as np
import os
from torch.utils.data import Dataset
from ctcdecode import CTCBeamDecoder
import Levenshtein as Lev
from hw3model import *
from hw3dataloader import *
from phoneme_list import *

best_accuracy = 0  # threshold to determine if a model checkpoint is formed
start_epoch = 0  # last checkpoint epoch

'''Load the data'''
features_dev = np.load('wsj0_dev.npy', allow_pickle=True)
labels_dev = np.load('wsj0_dev_merged_labels.npy', allow_pickle=True)
features_train = np.load('wsj0_train', allow_pickle=True)
labels_train = np.load('wsj0_train_merged_labels.npy', allow_pickle=True)
features_test = np.load('wsj0_test', allow_pickle=True)


'''This function trains my model. Iterates over the number of epochs and uses the training data loader, and outputs the
   training error. It then tests the model over the validation data if I have the function not commented out.'''
def train(model, training_data_loader, validation_data_loader, test_data_loader, criterion, optimizer, num_epochs, device, log_interval, scheduler):
    model.train()  # initialize training of model
    global best_accuracy
    global start_epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        torch.cuda.empty_cache()
        for batch_idx, (data, target, X_train_lens, Y_train_lens) in enumerate(training_data_loader):  # loop over the training data
            optimizer.zero_grad()  # clear gradients
            data = data.to(device)  # make data run on device (GPU or CPU)
            target = target.to(device)  # make target data(labels) run on device
            X_train_lens = X_train_lens.to(device) #make input data lengths run on device
            target_lengths = Y_train_lens.to(device) #make label data lengtsh run on device
            outputs, output_lengths = model(data,X_train_lens)  # forward pass through model to get predictions from our data(features)
            loss = criterion(outputs, target, output_lengths,target_lengths)  # get loss from forward pass and the target values
            running_loss += loss.item()  # sum the loss
            loss.backward()  # backward pass through to calculate weights and bias updates needed
            optimizer.step()  # step and update weights to linear and BN layers

            if batch_idx % log_interval == 0:
                print('Start Epoch: {} Train Epoch: {} Batch Number: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    start_epoch, epoch + 1, batch_idx, batch_idx * len(data), len(training_data_loader.dataset),
                                 100. * batch_idx / len(training_data_loader), loss.item()))
            del data  #delete the variables stored in cuda to give enough RAM for calculations
            del target
            del X_train_lens
            del target_lengths
            torch.cuda.empty_cache()
        end_time = time.time()
        running_loss /= len(training_data_loader)  # calculate running loss
        print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
        scheduler.step()
        test(model, validation_data_loader, criterion, device) #test the validaton data to see how model is doing on unseen data
        # print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(val_loss, val_acc))
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('ConvLSTM5_model_checkpoint'):
            os.mkdir('ConvLSTM5_model_checkpoint')
        torch.save(state, './ConvLSTM5_model_checkpoint/ConvLSTM5_checkpoint.pth')

'''Test function tests the model on validation data.'''
def test(model, dev_data_loader, criterion, device):
    global best_accuracy
    with torch.no_grad():
        model.eval() #get model into evaluation mode to remove BN and dropout layers
        running_loss = 0.0
        total_predictions = 0.0 #for calculating number of total predictions made
        torch.cuda.empty_cache()
        decoder = CTCBeamDecoder(PHONEME_MAP + [' '], beam_width=40,blank_id=0, log_probs_input=True) #initialize the decoder
        for batch_idx, (data, target, X_dev_lens, Y_dev_lens) in enumerate(dev_data_loader): #load the validation data
            data = data.to(device)
            target = target.to(device)
            X_dev_lens = X_dev_lens.to(device)
            target_lengths = Y_dev_lens.to(device)
            outputs, output_lengths = model(data, X_dev_lens)
            total_predictions += target.size(0)  # calculate number of predictions made
            loss = criterion(outputs, target, output_lengths, target_lengths).detach()  # calculate loss from torch.nn.CTCLoss()
            running_loss += loss.item()
            test_Y, _, _, test_Y_lens = decoder.decode(outputs.transpose(0, 1), output_lengths) #generate the decoded values
            char_err = []
            for i in range(test_Y.size(0)):
                prediction_value = " ".join(PHONEME_MAP[o] for o in test_Y[i, 0, :test_Y_lens[i, 0]]) #get the best value (from recitation 8)
                prediction_value = prediction_value.replace(' ', '') #removed empty space to get phonemes without weird/strange spaces to compare
                true_value = "".join(PHONEME_MAP[l] for l in target[i]) #finding true value from the targets
                ls = Lev.distance(prediction_value.replace(' ', ''), true_value.replace('_','')) #removed _ to get rid of weird underscores taking away from proper character error calcs
                char_err.append(ls) #append ls (Levhenstein value of error, distance).
            del data
            del target
            del X_dev_lens
            del target_lengths
            torch.cuda.empty_cache()
        running_loss /= len(dev_data_loader)
        character_error = np.sum(np.asarray(char_err))
        char_err_val = (character_error / total_predictions) * 100.0
        print('Testing Loss: ', running_loss)
        print('Character Error: ', char_err_val, '%')
        model.train()
        return running_loss, char_err_val

'''Acquire predictions for unseen data with no labels.'''
def test_predictions(model, test_data_loader, criterion, device):
    predictions = []
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        decoder = CTCBeamDecoder(PHONEME_MAP + [' '], beam_width=100,blank_id=0, log_probs_input=True) #initialize decoder with larger beam width (more accurate)
        for batch_idx, (data, X_test_lens) in enumerate(test_data_loader): #iterate over the test data
            data = data.to(device)
            X_test_lens = X_test_lens.to(device)
            outputs, output_lengths = model(data, X_test_lens)
            test, _, _, test_lengths = decoder.decode(outputs.transpose(0, 1), output_lengths) #generate the decoded values
            for i in range(test.size(0)):
                prediction_values = "".join(PHONEME_MAP[o] for o in test[i, 0, :test_lengths[i, 0]]) #get best prediction values
                prediction_values = prediction_values.replace(' ', '')
                predictions.append(prediction_values)
            del data
            del X_test_lens
        predictions_final = np.asarray(predictions).T #create predictions array for column stacking to output CSV
        columns = np.asarray(np.asmatrix(range(len(predictions_final))).T)  #column indices
        data_stuff = np.column_stack((columns, predictions_final))
        np.savetxt('Kaggle_Submission9HW3.csv', data_stuff, delimiter=",", fmt='%s', header="Id,Predicted", comments='') #save a CSV file

def main():
    global best_accuracy
    global start_epoch

    lr = 0.0001
    train_batch_size = 32
    no_cuda = False
    seed = 11785
    log_interval = 50
    resume = True

    cuda = not no_cuda and torch.cuda.is_available()  # check if GPU is available
    print(cuda)
    torch.manual_seed(seed)  # set seed manually
    device = torch.device("cuda:0" if cuda else "cpu")  # gives whether the device is CPU or GPU

    if cuda:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    '''Data loaders below call in the data set, and give other arguments like batch size, shuffle, num_workers, etc.'''
    training_data_loader = torch.utils.data.DataLoader(TrainData(features_train, labels_train), batch_size=train_batch_size, shuffle=True, collate_fn=collate_train, **kwargs)
    dev_data_loader = torch.utils.data.DataLoader(TrainData(features_dev, labels_dev), batch_size=train_batch_size, shuffle=True, collate_fn=collate_train, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(TestData(features_test), batch_size=train_batch_size, shuffle=False, collate_fn=collate_test, **kwargs)

    embedding_size = 40 #embedding size for phonemes
    channel_size = 128
    hidden_size = 256
    model = MyModel(embedding_size, channel_size, hidden_size, stride=1)
    model.apply(init_weights)
    criterion = torch.nn.CTCLoss() #CTC loss criterion
    Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=2, gamma=0.92) #use a learning rate scheduler to make sure model keeps learning
    model.train()
    model.to(device)
    print(model)
    ''' To resume model after training from a checkpoint in order to test different lrs manually or get predictions after training. '''
    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('ConvLSTM5_model_checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ConvLSTM5_model_checkpoint/ConvLSTM5_checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

    n_epochs = 27  # set num epochs
    torch.cuda.empty_cache()
    train(model, training_data_loader, dev_data_loader, test_data_loader, criterion, Optimizer, n_epochs, device, log_interval, scheduler)  # train model
    #test(model, dev_data_loader, criterion, device)
    #test_predictions(model, test_data_loader, criterion, device)

if __name__ == '__main__':
    main()