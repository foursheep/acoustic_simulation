import torch
from model import NetBackwardTOF, NetBackward, NetForward

from torchsummary import summary

# net = NetBackwardTOF().to('cuda')
# summary(net,(64,88,88),128,'cuda')

net = NetForward().to('cuda')
summary(net, (1,88*88))


# net = NetBackward()
# summary(net, (64, 415), 128)
