#from __future__ import print_function
import argparse
import torch
import torch.nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import itertools
import os
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from EfficientNetModified import *
from dataparser import *
from Closs import *
from myCNN import *
# from ResNet import *

best_accuracy = 0  # threshold to determine if a model checkpoint is formed
start_epoch = 0  # last checkpoint epoch
#
# #Load the datasets
# train_img_list, train_label_list, train_class_n = parse_data_train('train_data/medium')
# train_set = TrainingImageDataset(train_img_list, train_label_list)
#
# validation_img_list, validation_label_list, validation_class_n = parse_data_train('validation_classification/medium')
# valid_set = TrainingImageDataset(validation_img_list, validation_label_list)
#
#
test_img_list = parse_data_test('test_classification/medium')
test_set = TestImageDataset(test_img_list)
TRANSFORM_IMG = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    ])

#Load the datasets
# train_img_list, train_label_list, train_class_n = parse_data('train_data/medium')
# train_set = TrainingImageDataset(train_img_list, train_label_list)

training_imageFolder_dataset = torchvision.datasets.ImageFolder(root='train_data/medium', transform=TRANSFORM_IMG)

# validation_img_list, validation_label_list, validation_class_n = parse_data('validation_classification/medium')
# valid_set = TrainingImageDataset(validation_img_list, validation_label_list)

valid_imageFolder_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=TRANSFORM_IMG)
test_imageFolder_dataset = torchvision.datasets.ImageFolder(root= 'test_classification', transform = TRANSFORM_IMG)
'''Function used to train the model. Returns the running loss for calculation purposes in test function. This function was 
structured based on Recitation 1 code.'''

def train(args, model, training_data_loader, validation_data_loader, criterion, optimizer, num_epochs, device, task='Classification'):
    model.train()  #initialize training of model
    global best_accuracy
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(training_data_loader):  # loop over the training data
            optimizer.zero_grad()  # clear gradients
            data = data.to(device)  # make data run on device (GPU or CPU)
            target = target.to(device)  # make target data(labels) run on device
            outputs = model(data)  # forward pass through model to get predictions from our data(features)
            loss = criterion(outputs, target)  # get loss from forward pass and the target values
            running_loss += loss.item()  # sum the loss
            loss.backward()  # backward pass through to calculate weights and bias updates needed
            optimizer.step()  # step and update weights to linear and BN layers

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} Batch Number: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx, batch_idx * len(data), len(training_data_loader.dataset),
                                      100. * batch_idx / len(training_data_loader), loss.item()))

            # The information above was used as a percentage calculator to see the script was running through with some information as well

        end_time = time.time()
        running_loss /= len(training_data_loader)  # calculate running loss
        print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')

        if task == 'Classification':
            val_loss, val_acc = test_classify(model, validation_data_loader, criterion, device, epoch)
            print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(val_loss, val_acc))
            if val_acc > best_accuracy:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('ResNet_model_checkpoint'):
                    os.mkdir('ResNet_model_checkpoint')
                torch.save(state, './ResNet_model_checkpoint/ResNet_checkpoint.pth')
                best_accuracy = val_acc
        #scheduler.step()
        # else:
        #     test_verify(model, test_loader)

''' Test function gives us the relationship between our target output from our model and the given labels from the dataset. Modelled after
recitation 1 code'''
def test_classify(model, dev_data_loader, criterion, device, epoch):
    global best_accuracy
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
            total_predictions += target.size(0)  # calculate number of predictions made
            correct_predictions += (predicted == target).sum().item()  # sum number of correct predictions
            loss = criterion(outputs, target).detach()  # calculate loss from torch.nn.CrossEntropy
            running_loss += loss.item()
        running_loss /= len(dev_data_loader)
        acc = (correct_predictions / total_predictions) * 100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        model.train()
        return running_loss, acc

def test_verify(model, test_loader):
    raise NotImplementedError

'''This function is our label predictor once the model has been trained. Similar to function above but removes portions with labels'''
def test_classify_predictions(model, test_data_loader, device):
    model.eval()
    predictions = []
    predictions_final = []
    f = open('test_order_classification.txt', 'r')
    x = f.read().splitlines()
    f.close()
    image_names = np.asarray(np.asmatrix(np.asarray(x))).T
    # for batch_idx, data in enumerate(test_data_loader): #iterate through test data
    #     data = data.to(device)
    #     outputs = model(data)
    #     _, predicted = torch.max(outputs.data, 1) #acquire predicted values
    #     predictions.append(predicted.cpu().numpy().flatten()) #append predictions after turning to numpy and flattened
    for batch_num, (feats, labels) in enumerate(test_data_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        predictions.append(pred_labels.cpu().numpy().flatten())  # append predictions after turning to numpy and flattened
    predictions_final = np.asarray(np.asmatrix(list(itertools.chain.from_iterable(predictions))).T) #used to give us a column of our predictions
    predictions_final = [int(i) for i in predictions_final]
    predictions_final = np.asarray(predictions_final)
    final_mapping = []
    dict_test = training_imageFolder_dataset.class_to_idx
    dict_test_final = {value: key for key, value in dict_test.items()}
    for i in predictions_final:
        final_mapping.append(dict_test_final[i])
    final_mapping = np.asarray(np.asmatrix(final_mapping)).T
    data_stuff = np.column_stack((image_names, final_mapping))
    #columns = np.asarray(np.asmatrix(range(len(predictions_final))).T)  #column indices
    np.savetxt('HW2_P2_Classification_Kaggle_Submission1.csv', data_stuff, delimiter=",", fmt='%s', header="Id,Category", comments='') #save a csv file

def main():
    global best_accuracy
    global start_epoch
    global train_class_n
    #Training settings for the model(Simple place to modify some params)
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--train_batch_size', type=int, default=256, metavar='N',
                        help='Input batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training if true')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=250, metavar='LI',
                        help='number of batches to wait before logging')
    parser.add_argument('--resume', type=bool, default=True, metavar='RES',
                        help='Resume from a previous state model')
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()   #check if GPU is available
    torch.manual_seed(args.seed)        #set seed manually
    device = torch.device("cuda:0" if cuda else "cpu") #gives whether the device is CPU or GPU

    if cuda:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    '''Data loaders below call in the data set, and give other arguments like batch size, shuffle, num_workers, etc.'''
    # training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    # validation_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.train_batch_size, shuffle=False, **kwargs)
    '''Data loaders below call in the data set, and give other arguments like batch size, shuffle, num_workers, etc.'''
    training_imageFolder_dataloader = DataLoader(training_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    validation_imageFolder_dataloader = DataLoader(valid_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_imageFolder_dataloader = DataLoader(test_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=False, **kwargs)
    closs_weight = 1
    lr_cent = 0.5
    num_feats = 3
    learningRate = args.lr
    weightDecay = 5e-5
    feat_dim = 10
    # model = EfficientNetB0()
    model = ResNet34()
    model.apply(init_weights)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_label = torch.nn.CrossEntropyLoss()  # define criterion for loss
    # criterion_closs = CenterLoss(num_classes, feat_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)  # set optimizer and learning rate
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=5e-4, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    #optimizer_label = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    #optimizer_closs = optim.SGD(criterion_closs.parameters(), lr=lr_cent)
    model.train()
    model.to(device)
    print(model)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('ResNet_model_checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ResNet_model_checkpoint/ResNet_checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        best_accuracy = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    n_epochs = 6           #set num epochs
    #train(args, model, training_data_loader, validation_data_loader, criterion, optimizer, n_epochs, device) #train model
    # train_closs(model, training_data_loader, validation_data_loader, device, optimizer_label, optimizer_closs, criterion_label, criterion_closs, closs_weight, n_epochs, best_accuracy)
    train(args, model, training_imageFolder_dataloader, validation_imageFolder_dataloader, criterion, optimizer, n_epochs, device)
    test_classify_predictions(model, test_imageFolder_dataloader, device) #acquire predictions from trained model


if __name__ == '__main__':
    main()