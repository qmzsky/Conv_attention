import torch
import torch.nn as nn

class testNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)

    def forward(self,x):
        return

if __name__ == '__main__':
    net = testNet()
    lr = 0.01
    conv1_para = list(map(id,net.conv1.parameters()))