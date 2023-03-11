import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 


def fx(x, len, k):
    y = k - (x/len)*(2*k-1)
    return y

def makeMh(a, psize, k):
    #psize>=5, psize%2==1, mid>=2
    mid = int(psize/2)
    len = mid+1

    for i in range(psize):
        for j in range(psize):
            h = max(math.fabs(i-mid), math.fabs(j-mid))
            a[i][j] = fx(h, mid, k)
                
    return a

class HardCESA(nn.Module):
    def __init__(self, ksize, k):
        super(HardCESA, self).__init__()
        self.ksize = ksize
        self.w = nn.Parameter(torch.empty(ksize[-2], ksize[-1]),requires_grad=False) 
        self.w = makeMh(self.w, ksize[-1], k)

    def forward(self, x):
        psize = self.ksize[-1]
        inchannle = self.ksize[0]
        
        w = self.w
        w = w.reshape(1, 1, psize, psize)
        w = w.repeat(x.shape[0], inchannle, 1, 1)
        return w

class F3(nn.Module):
    def __init__(self, ksize):
        super(F3, self).__init__()
        self.ksize = ksize
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ksize[0],  1, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1))

    def forward(self, x):
        psize = self.ksize[-1]
        mid = int(psize/2)
        x = x[:,  :, mid-1:mid+2, mid-1:mid+2]

        x = self.conv(x)
        x = x.reshape(x.shape[0], x.shape[1])
        x = torch.sigmoid(x)
        x = x.reshape(x.shape[0],  x.shape[1], 1, 1)
        x = x.repeat(1, 1, psize, psize)
        return x

class SoftCESA(nn.Module):
    def __init__(self, ksize):
        super(SoftCESA, self).__init__()
        self.ksize = ksize
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(3, ksize[-1]*ksize[-1], kernel_size=(ksize[-1], ksize[-1]), stride=(1, 1), dilation=(1, 1))

    def forward(self, x, f3):
        psize = self.ksize[-1]
        inchannle = self.ksize[0]

        f1 = torch.max(x, dim=1)[0]
        f1 = f1.reshape(f1.shape[0], 1, f1.shape[1], f1.shape[2])
        f2 = torch.sum(x, dim=1)/inchannle
        f2 = f2.reshape(f2.shape[0], 1, f2.shape[1], f2.shape[2])
        f3 = f2*f3
        f = torch.cat((f1, f2, f3), dim=1)
        f = self.conv(f)
        f = f.reshape(f.shape[0], psize, psize)
        f = torch.sigmoid(f)
        f = f.reshape(f.shape[0], 1, psize, psize)
        f = f.repeat(1, inchannle, 1, 1)
        return f

class CESA(nn.Module):
    def __init__(self, ksizein, k):
        super(CESA, self).__init__()
        self.f3 = F3(ksizein)
        self.mh = HardCESA(ksizein, k)
        self.ms = SoftCESA(ksizein)

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ksizein[0], ksizein[0], kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1),padding=(1,1))
        self.bn = nn.BatchNorm2d(ksizein[0])


    def forward(self, x):
        f3 = self.f3(x)
        mh = self.mh(x)
        ms = self.ms(x, f3)
        m = ms + mh
        x = x*m
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    inputx = np.ones([16, 30, 9, 9])
    inputx = torch.from_numpy(inputx)
    inputx = inputx.float()
    net = CESA((30, 9, 9), 0.9)
    out = net(inputx)
    print(out.shape)
