import torch
from torch import nn
import seaborn as sns

# Setting a plot style.
sns.set(style="darkgrid")


class MesoNet4(nn.Module):

    """
    class holding the pytorch implementation of MesoNet: https://github.com/DariusAf/MesoNet
    """

    def __init__(self, dropout=0.5, device=None):
        super().__init__()

        self.dropout = dropout
        # convolution layers
        self.convs = nn.Sequential(

            # first block
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),

            # second block
            nn.Conv2d(8, 8, 5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),

            # third block
            nn.Conv2d(8, 16, 5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

            # fourth block
            nn.Conv2d(16, 16, 5, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4, 4),
        )

        self.linear = nn.Sequential(

            nn.Dropout(self.dropout),
            nn.Linear(784, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self.dropout),
            nn.Linear(16, 1),
            # do sigmoid in training script
        )

        # A placeholder for metric plots.
        self.metric_plots = dict()

    def forward(self, x):

        # push through convolution block
        x = self.convs(x)

        # flatten
        x = x.flatten(start_dim=1)

        # push trough linear block
        pred = self.linear(x)

        return pred

    def extract_features(self, x):
        return self.convs(x)


class Meso4LSTM(nn.Module):

    """
    class holding the MesoNet + LSTM model type
    """

    def __init__(self, seq_len=5, hidden_size=128, dropout=0.5, recurrent_dropout=0.2, device=None, gradcam_mode=False):
        super().__init__()

        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.gradcam_mode = gradcam_mode

        # convolution layers
        self.cnn = MesoNet4()

        self.rnn = nn.LSTM(input_size=784,
                          hidden_size=self.hidden_size,
                          num_layers=self.seq_len,
                          batch_first=True,
                          dropout=self.recurrent_dropout)

        self.linear = nn.Sequential(

            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self.dropout),
            nn.Linear(16, 1),
            # do sigmoid in training script
        )

        # A placeholder for metric plots.
        self.metric_plots = dict()

    def forward(self, x):
        if not self.gradcam_mode:
            # get dims
            batch, seq_len, c, h, w = x.size()

            # reformat for feedforward through CNN
            cnn_input = x.view(batch * seq_len, c, h, w)

            # push through conv block
            cnn_out = self.cnn.extract_features(cnn_input)

            # reformat for feed through RNN
            rnn_in = cnn_out.view(batch, seq_len, -1)

            # init LSTM
            h0 = torch.randn(self.seq_len, rnn_in.size(0), self.hidden_size).to(self.device)
            c0 = torch.randn(self.seq_len, rnn_in.size(0), self.hidden_size).to(self.device)

            # push through LSTM
            rnn_out, (h_n, c_n), = self.rnn(rnn_in, (h0, c0))

            # push through linear
            pred = self.linear(rnn_out[:, -1, :])

            return pred

        else:
            return self.cnn(x)
