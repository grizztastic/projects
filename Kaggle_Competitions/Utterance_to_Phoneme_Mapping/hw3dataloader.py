import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

'''Simple Dataloader for the training data that returns the input data and labels along with their lengths. Derived from recitation 8 code.'''
class TrainData(Dataset):
  def __init__(self, X, Y):
    self.X_train = X
    self.Y_train = Y
    self.X_train_lens = torch.LongTensor([len(seq) for seq in X])
    self.Y_train_lens = torch.LongTensor([len(seq) for seq in Y])

  def __len__(self):
    return len(self.X_train_lens)

  def __getitem__(self,index):
    X_train = self.X_train[index]
    Y_train = self.Y_train[index]
    X_train_lens = self.X_train_lens[index]
    Y_train_lens = self.Y_train_lens[index]
    return X_train, Y_train, X_train_lens, Y_train_lens

'''Custom training collate function to achieve tensors and proper batching for input data, labels, and their lengths'''
def collate_train(sequences):
    X_train = []
    X_train_lens = []
    Y_train = []
    Y_train_lens = []
    for seq in sequences:
        X_train.append(torch.from_numpy(seq[0]))
        X_train_lens.append(seq[2])
        Y_train.append(torch.from_numpy(seq[1]))
        Y_train_lens.append(seq[3])
    X_train = pad_sequence(X_train, batch_first=True) #pad the input data
    Y_train = pad_sequence(Y_train, batch_first=True) #pad the target data
    return X_train, Y_train, torch.LongTensor(X_train_lens), torch.LongTensor(Y_train_lens)

'''Simple Dataloader for the testing data that returns the input data and it's lengths. Derived from recitation 8 code.'''
class TestData(Dataset):
  def __init__(self, X):
    self.X_test = X
    self.X_test_lens = torch.LongTensor([len(seq) for seq in X])

  def __len__(self):
    return len(self.X_test_lens)

  def __getitem__(self,index):
    X_test = self.X_test[index]
    X_test_lens = self.X_test_lens[index]
    return X_test, X_test_lens

'''Custom testing collate function to achieve tensors and proper batching for input data its length'''
def collate_test(sequences):
    X_test = []
    X_test_lens = []
    for seq in sequences:
        X_test.append(torch.from_numpy(seq[0]))
        X_test_lens.append(seq[1])
    X_test = pad_sequence(X_test, batch_first=True) #pad the sequence
    return X_test, torch.LongTensor(X_test_lens)