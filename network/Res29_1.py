import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter as P
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4, with_norm=False):
        super(SEBasicBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU()
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residule = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.se(out)
        out += residule
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Conv - Relu - Conv - x
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = conv3x3(channels, channels, 1)
        self.relu_conv2 = conv3x3(channels, channels, 1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.relu_conv1(x))
        out = self.relu_conv2(out)
        out = out + residual
        return out
# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class DeepPoolLayer(nn.Module):
    def __init__(self, k, channels):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2, 4, 8]

        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.upsample_bilinear(y, x_size[2:]))
        resl = self.relu(resl)
        return resl

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y*x


class Attention(nn.Module):
  def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)

  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def __call__(self,x):
        out = self.body(x)
        return out + x


class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.b0_1 = nn.Conv2d(channels, channels//4, 1, 1, 0, bias=False)
        self.b0_2 = nn.Conv2d(channels, channels//4, 1, 1, 0, bias=False)
        self.b0_3 = nn.Conv2d(channels, channels//4, 1, 1, 0, bias=False)

        self.conv1_1 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 2, 2, bias=False),  nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 4, 4, bias=False),  nn.ReLU())
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 6, 6, bias=False),  nn.ReLU())
        self.conv4_1 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 8, 8, bias=False), nn.ReLU())

        self.conv1_2 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 2, 2, bias=False), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 4, 4, bias=False), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 6, 6, bias=False), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 8, 8, bias=False), nn.ReLU())

        self.conv1_3 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 2, 2, bias=False), nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 4, 4, bias=False), nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 6, 6, bias=False), nn.ReLU())
        self.conv4_3 = nn.Sequential(nn.Conv2d(channels//4, channels//4, 3, 1, 8, 8, bias=False), nn.ReLU())

        self.b_1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.b_4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()
    def __call__(self, x):
        residual = x
        x = self.relu(self.b0_1(x))
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1.append(self.conv4_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_1_ = self.relu(self.b0_2(buffer_1))
        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1_))
        buffer_2.append(self.conv2_2(buffer_1_))
        buffer_2.append(self.conv3_2(buffer_1_))
        buffer_2.append(self.conv4_2(buffer_1_))

        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_2_ = self.relu(self.b0_3(buffer_2))
        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2_))
        buffer_3.append(self.conv2_3(buffer_2_))
        buffer_3.append(self.conv3_3(buffer_2_))
        buffer_3.append(self.conv4_3(buffer_2_))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return residual + buffer_1 + buffer_2 + buffer_3


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResB(64),
             ResB(64),
             ResASPPB(64),
             ResB(64),
             ResB(64),
             nn.Conv2d(channels, channels, 3, padding=1)
             ]
        )

    def forward(self, x):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = x

        for layer in self.recursive_unit:
            x = layer(x)

        out = F.relu(xs + x)

        return out


class transBlock(nn.Module):
    def __init__(self, inplanes=64+3, planes=64, stride=1, downsample=None, reduction=4, with_norm=False):
        super(transBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)
        #self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU()

    def forward(self, x, f1):
        resiudal = f1

        f = torch.cat([f1, x], 1)
        f = self.relu(self.conv1(f))
        f = self.conv2(f)

        return f + resiudal


class PaiDehaze(nn.Module):
    def __init__(self):
        super(PaiDehaze, self).__init__()
        self.channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        print('conv1',self.conv1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            ResB(64)
        )
        print('conv2',self.conv2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            ResB(64)
        )

        self.lm1 = MemoryBlock(self.channels, self.channels)
        self.lm2 = MemoryBlock(self.channels, self.channels)
        self.lm3 = MemoryBlock(self.channels, self.channels)
        self.lm4 = MemoryBlock(self.channels, self.channels)

        self.t1 = transBlock()
        self.t2 = transBlock()
        self.t3 = transBlock()
        self.t4 = transBlock()

        self.end_conv1 = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.channels, 2),
            nn.Softmax()
        )

        # Last layers
        # -- Up1 --
        self.upconv1 =nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        self.upsamp1 = nn.Sequential(
            nn.PixelShuffle(2),
            ResB(64)
        )

        # ----------
        self.conv4 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1)

        # -- Up2 --
        self.upconv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        self.upsamp2 = nn.Sequential(
            nn.PixelShuffle(2),
            ResB(64)
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, dense=True):
        residual = x
        x0 = F.avg_pool2d(x, 4, 4)
        xd = self.relu(self.conv1(x))
        xd1 = self.relu(self.conv2(xd))
        f0 = self.relu(self.conv3(xd1))

        f1 = self.lm1(f0)
        f1 = self.t1(x0, f1)
        f2 = self.lm2(f1)
        f2 = self.t2(x0, f2)
        f3 = self.lm3(f2)
        f3 = self.t3(x0, f3)
        f4 = self.lm3(f3)
        f4 = self.t4(x0, f4)

        f5 = self.upconv1(f4)
        f5 = self.upsamp1(f5)
        f5 = self.relu(self.conv4(torch.cat([xd1, f5], 1)))

        f5 = self.upconv2(f5)
        f5 = self.upsamp2(f5)

        out1R = self.tanh(self.end_conv1(torch.cat([xd, f5], 1)))

        out1 = residual + out1R

        return out1


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation = dilation
        if dilation == 1:
            reflect_padding = int(np.floor(kernel_size / 2))
            self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, padding=dilation)

    def forward(self, x):
        if self.dilation == 1:
            out = self.reflection_pad(x)
            out = self.conv2d(out)
        else:
            out = self.conv2d(x)
        return out


class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.model1 = PaiDehaze().cuda()
       
        self.model1.load_state_dict(torch.load("/home/ywj/game/models/modelOut/MS57/Dem_60.pkl"))

        for p in self.model1.parameters():
            p.requires_grad = False

        self.model2 = PaiDehaze().cuda()
        self.model2.load_state_dict(torch.load("/home/zcc/Samples/broken/game-new/models/modelOut/MS56/Dem_60.pkl"))
        for p in self.model2.parameters():
            p.requires_grad = False

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResB(32),
            ResB(32),
            ResB(32),
            ResB(32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(F.interpolate(x, scale_factor=0.5))

        out2 = F.interpolate(out2, scale_factor=2)
        x_f = torch.cat((out1, out2), 1)

        out = self.fuse(x_f)

        return out
