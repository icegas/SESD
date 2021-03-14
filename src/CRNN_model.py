import torch
from torch import nn
import torchaudio
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, n_mels=64, log_input=True, num_classes=5, sample_rate=16000, melfb=None, **kwargs):
        super(CRNN, self).__init__()

        self.n_mels = n_mels
        self.log_input = log_input

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=melfb["n_fft"], 
        win_length=melfb["win_length"], hop_length=melfb["hop_len"], 
        window_fn=torch.hamming_window, n_mels=n_mels)

        filters = [64, 64, 64]
        poolings = [(4, 1), (4, 1), (4, 1)]
        self.convs = []

        self.convs.append(self.conv_bn_dropout(
            1, filters[0], 3, 0.05, poolings[0]
        ) )

        for i in range(1, len(filters)):
            self.convs.append(self.conv_bn_dropout(
                filters[i-1], filters[i], 3, 0.05, poolings[i]
            ))

        self.gru = nn.GRU(64, 64, 1, bidirectional=True)

        self.fc = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
    
    def conv_bn_dropout(self, in_c, out_c, k_s, drop_rate, pool_size):

        return nn.Sequential( 
            nn.Conv2d(in_c, out_c, kernel_size=k_s, bias=False, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Dropout2d(drop_rate),
        ).cuda()
    
    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()
        
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        
        x = x.squeeze(-2).permute(0, 2, 1)
        
        x, _ = self.gru(x)

        x = self.fc(x)
        x = self.sigmoid(x)
        return x


