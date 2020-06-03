import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

'''Designed model with 2 conv 1d layers going from embedding size of 40 to 128, 128 to 256. Then followed by 8 bidirectional LSTM layers, followed
   by 4 MLP layers descending in size until our num classes of 47.'''
class MyModel(nn.Module):
    def __init__(self, embedding_size, channel_size, hidden_size, stride=1):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(embedding_size, channel_size, kernel_size=1) #size of 40
        self.bn1 = nn.BatchNorm1d(channel_size)
        self.conv2 = nn.Conv1d(channel_size, hidden_size, kernel_size=1) #size of 128
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.tanh = nn.Hardtanh(inplace=True)
        self.lstm= nn.LSTM(hidden_size, hidden_size, bidirectional=True, num_layers=8) #256 to 256
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)  #512 to 256
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)  #256 to 128
        self.linear3 = nn.Linear(hidden_size//2, hidden_size//4) #128 to 64
        self.linear4 = nn.Linear(hidden_size//4, 47) #64 to 47

    def forward(self, X, lengths):
        X = X.permute(1, 2, 0) #permute the data in order for conv layers to work
        X = X.contiguous()
        out = self.tanh(self.bn1(self.conv1(X)))
        out = self.tanh(self.bn2(self.conv2(out)))
        out = out.permute(0, 2, 1) #permute the data once again so the lstm layers work
        packed_out = self.lstm(pack_padded_sequence(out, lengths, enforce_sorted=False))[0]
        out, out_lens = pad_packed_sequence(packed_out)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = self.dropout(out)
        out = self.linear4(out).log_softmax(2)
        return out, out_lens

''' Weight initialization for the model. Taken from recitation 6.'''
def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)

