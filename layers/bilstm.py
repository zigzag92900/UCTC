import torch as tc
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x, hidden=None):
        # Input
        #   x: [batch, time, dim]
        #   h: [layer*2, batch, hidden]
        # Output
        #   z: [batch, time, hidden]
        #   h: [batch, hidden]
        #   c: [batch, hidden]
        hidden = self.initHidden(x.size(0))
        y, (h, c) = self.lstm(x, hidden)
        z = tc.sum(y.view(y.size(0), y.size(1), 2,  self.hidden_size), 2)
        h = tc.sum(h.view(self.num_layers, 2, y.size(0), self.hidden_size), 1)
        c = tc.sum(c.view(self.num_layers, 2, y.size(0), self.hidden_size), 1)
        return z, (h[self.num_layers-1, :, :], c[self.num_layers-1, :, :])

    def initHidden(self, batch_size):
        if tc.cuda.is_available():
            return (
                nn.init.kaiming_normal_(
                    tc.Tensor(self.num_layers * 2, batch_size, self.hidden_size).cuda()),
                nn.init.kaiming_normal_(
                    tc.Tensor(self.num_layers * 2, batch_size, self.hidden_size).cuda()),
            )
        else:
            return (
                nn.init.kaiming_normal_(
                    tc.Tensor(self.num_layers * 2, batch_size, self.hidden_size)),
                nn.init.kaiming_normal_(
                    tc.Tensor(self.num_layers * 2, batch_size, self.hidden_size)),
            )
