import torch
import torch.nn as nn 

class TruckNN (nn.Module):
    """
    A modified version of the CNN model, adapted from the NVIDIA architecture.
    https://arxiv.org/pdf/1604.07316v1.pdf

    Total params: 578943

    TODO: Examine the necessity to implement time series prediction using Conv2dLSTM.
    TODO: Input shape validation --> now is set as H = 80, W = 240
    """

    def __init__(self):
        super(TruckNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),          # N x 3 x 80 x 240 -> N x 24 x 38 x 118
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),         # N x 24 x 38 x 118 -> N x 36 x 17 x 57
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),         # N x 36 x 17 x 57 -> N x 48 x 7 x 27
            nn.ELU(),
            nn.BatchNorm2d(48)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 64, 3),                   # N x 48 x 7 x 27 -> N x 64 x 5 x 25
            nn.ELU(),
            nn.Conv2d(64, 64, 3),                   # N x 64 x 5 x 25 -> N x 64 x 3 x 23
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.25)
        )
   
        self.fc = nn.Sequential(
            nn.Linear(4416, 100),                   # N x 4416 -> N x 100
            nn.ELU(),
            nn.Linear(100, 50),                     # N x 100 -> N x 50
            nn.ELU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 10),                      # N x 50 -> N x 10
            nn.ELU()
        )

        self.out = nn.Linear(10, 1)                 # N x 10 -> N x 1

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 80, 240)           # N x 3 x H x W, H = 80, W = 240

        x = self.conv2(self.conv1(x))               # N x 64 x 3 x 23

        # input dimension needs to be monitored
        x = x.view(x.size(0), 4416)                 # N x 64 x 3 x 23 -> N x 4416

        x = self.fc(x)                              # N x 10

        x = self.out(x)                             # N x 1

        return x


"""
y = TruckNN()
print(y)
print(sum(p.numel() for p in y.parameters() if p.requires_grad))
"""