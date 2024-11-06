import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=2,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.2
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 64, kernel_size=8, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(64),
            self.GELU,
            
            nn.Conv1d(64, 32, kernel_size=8, stride=1, bias=False, padding=0),
            nn.BatchNorm1d(32),
            self.GELU,
            
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            
            nn.Conv1d(128, 64, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            self.GELU,
            
            nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(32),
            self.GELU,
            
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 32
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class DeepFeatureNet(nn.Module):
    def __init__(self, input_dims, n_classes, use_dropout=True):
        super(DeepFeatureNet, self).__init__()

        self.use_dropout = use_dropout
        self.MRCNN=MRCNN(3)
        # Convolutional layers with small filter size
        
        self.dropout=nn.Dropout(0.2)
        # Final layers
        self.fc_concat = nn.Linear(160, 40)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
    def _conv1d_layer(self, in_channels, out_channels, filter_size, stride):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=filter_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Stream 1
        
        # Flatten
        concatenated=self.MRCNN(x)
        flattened = concatenated.view(concatenated.size(0), -1)
        
        return flattened

    def _apply_dropout(self, x):
        if self.use_dropout:
            return F.dropout(x, p=0.1, training=self.training)
        else:
            return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = DeepFeatureNet(1,32)
        self.cnn2 = DeepFeatureNet(1,32)

        sizes = [60] +[100,100]# [192] *2#3192
        print(sizes)
        # projector
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i+1], sizes[i + 1], bias=False))
        self.predictor= nn.Sequential(*layers)
        
        #self.projector = nn.Linear(32, 6, bias=False)
        self.lambd = 0.0051
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        #self.bn = nn.BatchNorm1d(6, affine=False)
    def forward(self, x1,x2, label=None, mode='train'):
        if mode != 'train':
           return self.cnn1(x1)
        
        z1 = self.projector(self.cnn1(x1))
        z1 = self.predictor(z1)
        z2 = self.projector(self.cnn2(x2))
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss