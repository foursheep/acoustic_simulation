# import torch
import torch
import torch.nn as nn
import torchvision


class NetBackward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.kWaveBackward = nn.Sequential(
        #     nn.Linear(64*415, 64*121),
        #     nn.ReLU(),
        #     nn.Linear(64*121, 88*88)
        # )
        self.kWaveBackward = nn.Linear(64*415, 64*121)

    def forward(self, x):

        return self.kWaveBackward(x)


class NetForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.kWaveForward = nn.Linear(64 * 121, 64 * 415)
        # self.kWaveForward = nn.Sequential(
        #     nn.Linear(88*88, 64*121),
        #     nn.ReLU(),
        #     nn.Linear(64*121, 64*415)
        # )

    def forward(self, x):
        return self.kWaveForward(x)


class NetBackwardTOF(nn.Module):
    def __init__(self):
        super().__init__()
        self.filterOnTOF = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.1),

            nn.Conv2d(in_channels=32, out_channels=8,
                      kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(8),
            # nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=1,
                      kernel_size=3, padding=1,
                      stride=1),
        )

    def forward(self, x):
        x = self.filterOnTOF(x)
        return x
