import os
from torch.utils.data import Dataset
from scipy.io import loadmat
from torch import Tensor, flatten
import torch


Recondescribe = loadmat('./ReconDescribe.mat')
SpeedofSound = Tensor([1500])
dt = Tensor(Recondescribe['dt'])
sensor_location_x = Tensor(Recondescribe['sensor_location_x'])
sensor_location_y = Tensor(Recondescribe['sensor_location_y'])
p0_recon_location_x = Tensor(Recondescribe['p0_recon_location_x'])
p0_recon_location_y = Tensor(Recondescribe['p0_recon_location_y'])


class kWaveDatasetTrain(Dataset):
    def __init__(self):
        self.L0path = './L0'
        self.P0path = './P0'
        self.L0Dirlist = os.listdir(self.L0path)
        self.P0Dirlist = os.listdir(self.P0path)

    def __getitem__(self, index) -> (Tensor, Tensor):
        L0 = loadmat(os.path.join(self.L0path, self.L0Dirlist[index]))['sensor_data']
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']
        L0, P0 = Tensor(L0), Tensor(P0)
        P0 = flatten(P0)
        L0 = flatten(L0)
        return L0, P0

    def __len__(self):
        return len(self.L0Dirlist)


class kWaveDatasetTest(Dataset):
    def __init__(self):
        self.L0path = './TestPitch/L0'
        self.P0path = './TestPitch/P0'
        self.L0Dirlist = os.listdir(self.L0path)
        self.P0Dirlist = os.listdir(self.P0path)

    def __getitem__(self, index) -> (Tensor, Tensor):
        L0 = loadmat(os.path.join(self.L0path, self.L0Dirlist[index]))['sensor_data']
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']
        L0, P0 = Tensor(L0), Tensor(P0)
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']

        return L0, P0

    def __len__(self):
        return len(self.L0Dirlist)


class TOFkWaveDatasetTrain(Dataset):
    def __init__(self):
        self.L0path = './L0'
        self.P0path = './P0'
        self.L0Dirlist = os.listdir(self.L0path)
        self.P0Dirlist = os.listdir(self.P0path)

    def __getitem__(self, index) -> (Tensor, Tensor):
        L0 = loadmat(os.path.join(self.L0path, self.L0Dirlist[index]))['sensor_data']
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']
        L0, P0 = Tensor(L0), Tensor(P0)

        TOF = torch.ones(64, 88, 88)

        for i in range(64):
            distance_x = p0_recon_location_x - sensor_location_x[0, i]
            distance_y = p0_recon_location_y - sensor_location_y[0, i]
            distance_xy = torch.sqrt(distance_y.pow(2) + distance_x.pow(2))
            distance_xy_time = distance_xy / SpeedofSound

            distance_xy_index = torch.floor(distance_xy_time / dt) - 1

            distance_xy_index = torch.where(distance_xy_index < 0,
                                            torch.tensor(0.0),
                                            distance_xy_index)

            distance_xy_index = distance_xy_index.int()

            L0_iter = L0[i, :]

            TOF[i, :, :] = L0_iter[distance_xy_index]

        return TOF, P0

    def __len__(self):
        return len(self.L0Dirlist)


class TOFkWaveDatasetTest(Dataset):
    def __init__(self):
        self.L0path = './TestPitch/L0'
        self.P0path = './TestPitch/P0'
        self.L0Dirlist = os.listdir(self.L0path)
        self.P0Dirlist = os.listdir(self.P0path)

    def __getitem__(self, index) -> (Tensor, Tensor):
        L0 = loadmat(os.path.join(self.L0path, self.L0Dirlist[index]))['sensor_data']
        P0 = loadmat(os.path.join(self.P0path, self.P0Dirlist[index]))['p0']
        L0, P0 = Tensor(L0), Tensor(P0)

        TOF = torch.ones(64, 88, 88)

        for i in range(64):
            distance_x = p0_recon_location_x - sensor_location_x[0, i]
            distance_y = p0_recon_location_y - sensor_location_y[0, i]
            distance_xy = torch.sqrt(distance_y.pow(2) + distance_x.pow(2))
            distance_xy_time = distance_xy / SpeedofSound

            distance_xy_index = torch.floor(distance_xy_time / dt) - 1

            distance_xy_index = torch.where(distance_xy_index < 0,
                                            torch.tensor(0.0),
                                            distance_xy_index)

            distance_xy_index = distance_xy_index.int()

            L0_iter = L0[i, :]

            TOF[i, :, :] = L0_iter[distance_xy_index]

        return TOF, P0

    def __len__(self):
        return len(self.L0Dirlist)
