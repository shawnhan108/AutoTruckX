import torch
import torch.nn as nn 
from torchvision.models import resnet50

from utils import freeze_params

class TruckNN (nn.Module):
    """
    A modified version of the CNN model, adapted from the NVIDIA architecture.
    https://arxiv.org/pdf/1604.07316v1.pdf

    5 Conv2D layers + 3 dense layers. Total params: 0.6M (578941)

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


class TruckRNN (nn.Module):
    """
    A CNN conv3d + LSTM model, modified and based on the architecture defined in https://github.com/FangLintao/Self-Driving-Car.

    5 Conv3D layers with residual learning, 2 LSTMs. Total params: 1.7M (1738531)

    TODO: Input shape validation --> now is set as H = 80, W = 240, frame_num = D = 15
    """

    def __init__(self):
        super(TruckRNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(3, 5, 5), stride=(1, 2, 2)),                      # N x 3 x 15 x 80 x 240 -> N x 24 x 13 x 38 x 118
            nn.ELU(),
            nn.BatchNorm3d(24),
            
            nn.Conv3d(24, 36, kernel_size=(3, 5, 5), stride=(1, 2, 2)),                     # N x 24 x 13 x 38 x 118 -> N x 36 x 11 x 17 x 57
            nn.ELU(),
            nn.BatchNorm3d(36),

            nn.Conv3d(36, 48, kernel_size=(3, 5, 5), stride=(1, 2, 2)),                     # N x 36 x 11 x 17 x 57 -> N x 48 x 9 x 7 x 27
            nn.ELU(),
            nn.BatchNorm3d(48),
            
            nn.Conv3d(48, 64, kernel_size=3, stride=1),                                     # N x 48 x 9 x 7 x 27 -> N x 64 x 7 x 5 x 25
            nn.ELU(),
            nn.BatchNorm3d(64),
        )

        self.ResConv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=(1,1,1)),                    # N x 64 x 7 x 5 x 25 -> N x 64 x 7 x 5 x 25
            nn.ELU(),
            nn.BatchNorm3d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1),                                     # N x 64 x 7 x 5 x 25 -> N x 64 x 5 x 3 x 23
            nn.ELU(),
            nn.BatchNorm3d(64)
        )

        self.convActiv = nn.ELU()
        self.rnnActiv = nn.Tanh()
        self.flat = nn.Flatten(start_dim=2)

        self.lstm1 = nn.LSTM(input_size = 4416, hidden_size = 64, batch_first=True)         # N x 5 x 4416 -> N x 5 x 64
        self.lstm2 = nn.LSTM(input_size = 64, hidden_size = 32, batch_first=True)           # N x 5 x 64 -> N x 5 x 32

        self.fc = nn.Sequential(
            nn.Linear(32, 512),                                                             # 5 x N x 32 -> 5 x N x 512
            nn.ELU(),
            nn.Linear(512, 100),                                                            # 5 x N x 512 -> 5 x N x 100
            nn.ELU(),
            nn.Linear(100, 50),                                                             # 5 x N x 100 -> 5 x N x 50
            nn.ELU(),
            nn.Linear(50, 10),                                                              # 5 x N x 50 -> 5 x N x 10
            nn.ELU()
        )

        self.out = nn.Linear(10, 1)                                                         # 5 x N x 10 -> 5 x N x 1

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 15, 80, 240)                                               # N x 3 x D x H x W, D = 15, H = 80, W = 240

        # Convolution and Residual Learning
        x = self.conv1(x)                                                                   # N x 3 x 15 x 80 x 240 -> N x 64 x 7 x 5 x 25 

        residual = x                                                                        # N x 64 x 7 x 5 x 25 
        newx = self.ResConv(x)                                                              # N x 64 x 7 x 5 x 25
        x = self.convActiv(residual + newx)                                                 # N x 64 x 7 x 5 x 25

        x = self.conv2(x)                                                                   # N x 64 x 5 x 3 x 23

        del residual
        del newx

        # Recurrent, input dimension needs to be monitored
        x = x.permute([0,2,1,3,4])                                                          # N x 5 x 64 x 3 x 23
        x = self.flat(x)                                                                    # N x 5 x 64 x 3 x 23 -> N x 5 x 4416
        x = self.rnnActiv(self.lstm1(x)[0])                                                 # Extract output only. N x 5 x 64
        x = self.rnnActiv(self.lstm2(x)[0])                                                 # Extract output only. N x 5 x 32

        # Fully connected, input dimension needs to be monitored
        x = x.permute([1,0,2])                                                              # N x 5 x 32 -> 5 x N x 32
        x = self.fc(x)                                                                      # 5 x N x 32 -> 5 x N x 10

        x = self.out(x)                                                                     # 5 x N x 1

        return x.squeeze().permute(1, 0)                                                    # 5 x N x 1 -> N x 5


class TruckResnet50(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet50 for features extraction https://arxiv.org/abs/1512.00567

    Transfer Learning from pretrained Resnet-50, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)

    TODO: Input shape validation --> now is set as H = 224, W = 224
    """

    def __init__(self):
        super(TruckResnet50, self).__init__()

        self.resnet50 = resnet50(pretrained=True)
        freeze_params(self.resnet50)
        self.resnet50.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet50(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x


# """
# y = TruckResnet50()
# print(y)
# print(sum(p.numel() for p in y.parameters() if not p.requires_grad))
# x = torch.randn(2, 3, 224, 224)
# print(x.size())
# x = y(x)
# print(x.size())
# """
