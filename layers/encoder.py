import torch as tc
import numpy as npy
import torch.nn as nn
from layers.bilstm import BiLSTM


class CnnEncoder(nn.Module):
    # Input
    #   x: [batch, time, dim_in]
    # Output
    #   h: [batch, dim_out]
    def __init__(self, dim_in, dim_out, kernel=[3, 3, 3, 3, 3], hidden=[32, 32, 32, 32, 32]):
        super(CnnEncoder, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.cnn = nn.Sequential()
        num_layers = len(kernel)
        hidden.append(dim_in)
        for i in range(num_layers):
            self.cnn.add_module(f'conv{i}', nn.Conv1d(in_channels=hidden[i-1],
                                                      out_channels=hidden[i],
                                                      kernel_size=kernel[i],
                                                      stride=1,
                                                      padding=kernel[i]//2),)
            self.cnn.add_module(f'bn{i}', nn.BatchNorm1d(hidden[i]))
            self.cnn.add_module(f'relu{i}', nn.LeakyReLU(0.3))
        self.dnn = nn.Sequential(
            nn.Linear(in_features=hidden[num_layers-1], out_features=dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Softmax(dim=1),
        )
        self.ModelInit()

    def ModelInit(self):
        for layer in self.modules():
            if isinstance(layer, tc.nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, tc.nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x):
        # batch x time x dim
        seq = self.cnn(x.permute(0, 2, 1))
        # batch x dim x time
        embedded = tc.mean(seq, 2)
        # batch x dim
        embedded = self.dnn(embedded)
        return embedded


class LstmEncoder(nn.Module):
    # Input
    #   x: [batch, time, dim_in]
    # Output
    #   h: [batch, dim_out]
    def __init__(self, dim_in, dim_out, hidden=64, num_layers=3):
        super(LstmEncoder, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.rnn = BiLSTM(
            input_size=dim_in,
            hidden_size=hidden,
            num_layers=num_layers,
        )
        # self.pool = nn.AvgPool1d(time_steps)
        self.dnn = nn.Sequential(
            nn.Linear(in_features=hidden, out_features=dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Softmax(dim=1),
        )
        self.ModelInit()

    def ModelInit(self):
        for layer in self.modules():
            if isinstance(layer, tc.nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, tc.nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x):
        # batch x time x dim
        seq, (hn, cn) = self.rnn(x)
        # batch x  time x dim
        embedded = hn
        # batch x dim
        embedded = self.dnn(embedded)
        return embedded


class CnnLstmEncoder(nn.Module):
    # Input
    #   x: [batch, time, dim_in]
    # Output
    #   h: [batch, dim_out]
    def __init__(self, dim_in, dim_out, kernel=[3, 3, 3], hidden=[32, 32, 32, 32]):
        super(CnnLstmEncoder, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=dim_in,
                      out_channels=hidden[0],
                      kernel_size=kernel[0],
                      stride=1,
                      padding=kernel[0]//2),
            nn.BatchNorm1d(hidden[0]),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=hidden[0],
                      out_channels=hidden[1],
                      kernel_size=kernel[1],
                      stride=1,
                      padding=kernel[1]//2),
            nn.BatchNorm1d(hidden[1]),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=hidden[1],
                      out_channels=hidden[2],
                      kernel_size=kernel[2],
                      stride=1,
                      padding=kernel[2]//2),
            nn.BatchNorm1d(hidden[2]),
            nn.LeakyReLU(0.3),
        )
        self.rnn = BiLSTM(
            input_size=hidden[2],
            hidden_size=hidden[3],
            num_layers=1,
        )
        self.dnn = nn.Sequential(
            nn.Linear(hidden[3], dim_out, bias=False),
            nn.BatchNorm1d(dim_out),
            nn.Softmax(dim=1),
        )
        self.ModelInit()

    def ModelInit(self):
        for layer in self.modules():
            if isinstance(layer, tc.nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, tc.nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x):
        seq = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        # batch x time x dim
        seq, (hn, cn) = self.rnn(seq)
        embedded = tc.mean(seq, 1)
        # batch x dim
        embedded = self.dnn(embedded)
        return embedded


class CatEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=[3, 3, 3], hidden=[32, 32, 32, 32]):
        super(CatEncoder, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=dim_in,
                      out_channels=hidden[0],
                      kernel_size=kernel[0],
                      stride=1,
                      padding=kernel[0]//2),
            nn.BatchNorm1d(hidden[0]),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=hidden[0],
                      out_channels=hidden[1],
                      kernel_size=kernel[1],
                      stride=1,
                      padding=kernel[1]//2),
            nn.BatchNorm1d(hidden[1]),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=hidden[1],
                      out_channels=hidden[2],
                      kernel_size=kernel[2],
                      stride=1,
                      padding=kernel[2]//2),
            nn.BatchNorm1d(hidden[2]),
            nn.LeakyReLU(0.3),
        )
        self.rnn = BiLSTM(
            input_size=dim_in,
            hidden_size=hidden[3],
            num_layers=2,
        )
        self.dnn = nn.Sequential(
            nn.Linear(hidden[2]+hidden[3], dim_out, bias=False),
            nn.BatchNorm1d(dim_out),
            nn.Softmax(dim=1),
        )
        self.ModelInit()

    def ModelInit(self):
        for layer in self.modules():
            if isinstance(layer, tc.nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, tc.nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0.0)

    def forward(self, x):
        a = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        a = tc.mean(a, 1)
        # batch x time x dim
        b, (hn, cn) = self.rnn(x)
        b = tc.mean(b, 1)
        embedded = tc.cat([a, b], 1)
        # print(embedded.shape)
        # batch x dim
        embedded = self.dnn(embedded)
        return embedded
