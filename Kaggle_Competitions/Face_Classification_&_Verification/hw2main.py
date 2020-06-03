from __future__ import print_function
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
from torchvision.transforms import RandomHorizontalFlip, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from myCNN import *
from dataparser import *

best_accuracy = 0  # threshold to determine if a model checkpoint is formed
start_epoch = 0  # last checkpoint epoch

'''Transforms for all images to undergo during classification'''
TRANSFORM_IMG = transforms.Compose([
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

'''Use torchvision dataloader to read in all of the training data and test data for classification'''
training_imageFolder_dataset = torchvision.datasets.ImageFolder(root='train_data/medium', transform=TRANSFORM_IMG)
valid_imageFolder_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=TRANSFORM_IMG)
test_imageFolder_dataset = torchvision.datasets.ImageFolder(root='test_classification', transform = TRANSFORM_IMG)

f = open('test_trials_verification_student.txt', 'r')
x = f.read().splitlines()
f.close()
image_names_verification = np.asarray(np.asmatrix(np.asarray(x))).T  #get the image names for verification into a column
ordered_list = []
root = 'test_verification/'
for i in range(len(image_names_verification)):  #creates an ordered list of the images to read in from the specified folder.
    ordered_list.append(root + x[i].split(" ")[0])
    ordered_list.append(root + x[i].split(" ")[1])

test_verifcation_set = TestImageDataset(ordered_list) #Read in the data for verificaiton testing since there is no subfolder

'''This function trains my model and outputs a csv of test predictions along with saving the model state dict if the new accuracy
   is better than the previous epochs accuracy. Code was very similar to my hw1 code except I added the torch loader'''
def train(args, model, training_data_loader, validation_data_loader, test_data_loader, criterion, optimizer, num_epochs, device, scheduler, task='Classification'):
    model.train()  #initialize training of model
    global best_accuracy
    global start_epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(training_data_loader):  # loop over the training data
            optimizer.zero_grad()  # clear gradients
            data = data.to(device)  # make data run on device (GPU or CPU)
            target = target.to(device)  # make target data(labels) run on device
            outputs = model(data)[1]  # forward pass through model to get predictions from our data(features)
            loss = criterion(outputs, target)  # get loss from forward pass and the target values
            running_loss += loss.item()  # sum the loss
            loss.backward()  # backward pass through to calculate weights and bias updates needed
            optimizer.step()  # step and update weights to linear and BN layers

            if batch_idx % args.log_interval == 0:
                print('Start Epoch: {} Train Epoch: {} Batch Number: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    start_epoch, epoch + 1, batch_idx, batch_idx * len(data), len(training_data_loader.dataset),
                                      100. * batch_idx / len(training_data_loader), loss.item()))
            del data
            del target

        end_time = time.time()
        running_loss /= len(training_data_loader)  # calculate running loss
        print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
        scheduler.step()
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
                test_classify_predictions(model, test_data_loader, device)  # acquire predictions from trained model

''' Test Classify function gives us the relationship between our target output from our model and the given labels from the dataset. Modeled after
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
            outputs = model(data)[1]    #get outputs from the model
            _, predicted = torch.max(outputs.data, 1) #acquire prediction values
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

'''Function created to test the verification of my model'''
def test_verify(model, test_verifcation_dataloader, device, image_names):
    model.eval()
    Closs_list = []
    for batch_num, data in enumerate(test_verifcation_dataloader):
        data = data.to(device)
        Closs = model(data)[0]
        Closs_list.append(Closs.cpu().detach().numpy()) #append Closs values after turning to numpy and flattened
        del data
    
    Closs_list = np.asarray(np.asmatrix(np.asarray(list(itertools.chain.from_iterable(Closs_list)))))
    print(Closs_list.shape)
    Closs1 = [Closs_list[i] for i in range(len(Closs_list)) if i % 2 == 0] #split data into 2 lists to compare proper pictures
    Closs2 = [Closs_list[i] for i in range(len(Closs_list)) if i % 2 == 1]
    print(len(Closs1))
    print(len(Closs2))
    Closslen = len(np.asarray(Closs1))
    Cos_output = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)  #define cosine similarity metric
    for i in range(Closslen):
        C1 = torch.from_numpy(Closs1[i])  #make the values a torch tensor
        C2 = torch.from_numpy(Closs2[i])
        Cos_output.append(cos(torch.abs(C1), torch.abs(C2)).cpu().numpy().flatten())  #append cosine similarity results to list
        del C1
        del C2
    cos_mean_values_list = [np.mean(x) for x in Cos_output]  #acquire the mean value of each output
    cos_results = np.asmatrix(cos_mean_values_list)
    cos_results[cos_results < 0.5] = 0 #determine whether it is not similar
    cos_results[cos_results > 0] = 1  #or whether it is similar
    cos_results = np.asarray(cos_results).T
    cos_results = [int(i) for i in cos_results]
    data_stuff = np.column_stack((image_names, cos_results))
    np.savetxt('HW2_P2_Verification_Kaggle_Submission_ResNet.csv', data_stuff, delimiter=",", fmt='%s', header="trial,score", comments='')  # save
    model.train()

'''This function is our label predictor once the model has been trained. Similar to function above but removes portions with labels'''
def test_classify_predictions(model, test_data_loader, device):
    model.eval()
    predictions = []
    predictions_final = []
    f = open('test_order_classification.txt', 'r')
    x = f.read().splitlines()
    f.close()
    image_names = np.asarray(np.asmatrix(np.asarray(x))).T #image names column
    for batch_num, (feats, labels) in enumerate(test_data_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]  #get outputs from model
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)  #acquire predictions
        pred_labels = pred_labels.view(-1)
        predictions.append(pred_labels.cpu().numpy().flatten()) #append predictions after turning to numpy and flattened
    predictions_final = np.asarray(np.asmatrix(list(itertools.chain.from_iterable(predictions))).T) #used to give us a column of our predictions
    predictions_final = [int(i) for i in predictions_final]
    predictions_final = np.asarray(predictions_final)
    final_mapping = []
    dict_test = training_imageFolder_dataset.class_to_idx  #use built in class to idx function from torchvision to get inverse mapping
    dict_test_final = {value: key for key, value in dict_test.items()} #reverse the mapping to be idx to class
    for j in predictions_final:
        final_mapping.append(dict_test_final[j])
    final_mapping = np.asarray(np.asmatrix(final_mapping)).T
    data_stuff = np.column_stack((image_names, final_mapping))
    np.savetxt('HW2_P2_Classification_Kaggle_Submission_ResNet.csv', data_stuff, delimiter=",", fmt='%s', header="Id,Category", comments='') #save a csv file

def main():
    global best_accuracy
    global start_epoch

    #Training settings for the model(Simple place to modify some params)
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--train_batch_size', type=int, default=256, metavar='N',
                        help='Input batch size for training (default: 64)')
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
    training_imageFolder_dataloader = DataLoader(training_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    validation_imageFolder_dataloader = DataLoader(valid_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_imageFolder_dataloader = DataLoader(test_imageFolder_dataset, batch_size=args.train_batch_size, shuffle=False, **kwargs)
    test_verification_dataloader = torch.utils.data.DataLoader(test_verifcation_set, batch_size=args.train_batch_size, shuffle=False, **kwargs)

    model = ResNet34()
    model.apply(init_weights)
    criterion = torch.nn.CrossEntropyLoss()
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=1, gamma=0.9)
    model.train()
    model.to(device)
    print(model)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('ResNet_model_checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ResNet_model_checkpoint/ResNet_checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        best_accuracy = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    n_epochs = 10   #set num epochs
    train(args, model, training_imageFolder_dataloader, validation_imageFolder_dataloader, test_imageFolder_dataloader, criterion, Optimizer, n_epochs, device, scheduler)  # train model
    test_verify(model, test_verification_dataloader, device, image_names_verification)

if __name__ == '__main__':
    main()
