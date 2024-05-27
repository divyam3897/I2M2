"""Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
from torch._C import _ImperativeEngine
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from fastmri.data.transforms import to_tensor
from fastmri.fftc import ifft2c_new
from einops.layers.torch import Reduce

def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='group'):
        super(PreActBlock, self).__init__()
        if norm == 'group' :
            self.bn1 = nn.GroupNorm(32, in_planes)
        elif norm == 'layer':
            self.bn1 = nn.GroupNorm(1, in_planes)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if norm == 'group' :
            self.bn2 = nn.GroupNorm(32, planes)
        elif norm == 'layer' :
            self.bn2 = nn.GroupNorm(1, planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        image_shape,
        data_space, 
        return_features=False,
        num_classes=2,
        drop_prob=0.5,
        input_channels=1,
        downsample=False,
        shared_conv=False,
        fc_layer_dim=100,
        norm = 'group'
    ):
        super(PreActResNet, self).__init__()
        self.input_channels = input_channels
        self.shared_conv = shared_conv
        self.fc_layer_dim = fc_layer_dim
        channel_scale_factor = {1 : 1, 5 : 1, 15:4}
        self.data_space = data_space
        self.downsample = downsample
        self.norm = norm
        if self.downsample and self.input_channels > 1 :
            self.downsampled_channels = 5 # Can try tuning this
            self.conv1x1 = nn.Conv2d(self.input_channels, self.downsampled_channels, kernel_size=1, stride=1)
            self.input_channels = self.downsampled_channels 
        
        if self.shared_conv :
            self.input_channels = 1

        self.in_planes = 64 * channel_scale_factor[self.input_channels]

        self.avgpool = nn.AdaptiveAvgPool2d((int(self.fc_layer_dim**0.5),int(self.fc_layer_dim**0.5))) # can try modifying this
        self.globalpool = Reduce('b co c h w -> b c h w', 'mean')
        self.globalpool2 = nn.Conv3d(15, 1, 1)

        if norm == 'group' :
            self.bn1 = nn.GroupNorm(32, 512)
        else :
            self.bn1 = nn.BatchNorm2d(512)        
        self.conv1 = nn.Conv2d(1*self.input_channels, 64*channel_scale_factor[self.input_channels], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_p = nn.Conv2d(2*self.input_channels, 64*channel_scale_factor[self.input_channels], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64 * channel_scale_factor[self.input_channels], num_blocks[0], stride=1, norm=self.norm)
        self.layer2 = self._make_layer(block, 128 * channel_scale_factor[self.input_channels], num_blocks[1], stride=2, norm = self.norm)
        self.layer3 = self._make_layer(block, 256 * channel_scale_factor[self.input_channels], num_blocks[2], stride=2, norm = self.norm)
        self.layer4 = self._make_layer(block, 512 * channel_scale_factor[self.input_channels], num_blocks[3], stride=2, norm = self.norm)

        self.image_shape = image_shape
        self.return_features = return_features

        in_dim = 512 * channel_scale_factor[self.input_channels] * block.expansion * self.fc_layer_dim

        self.linear_mtear = nn.Linear(in_dim, num_classes)
        self.linear_acl = nn.Linear(in_dim, num_classes)
        self.linear_abnormal = nn.Linear(in_dim, num_classes)
        self.linear_cartilage = nn.Linear(in_dim, num_classes)
        self.bn_2D = nn.BatchNorm2d(512)
        self.bn_3D = nn.BatchNorm3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride, norm='group'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

 
    def get_outputs(self, out, batch_size):
        layer_1_out = self.layer1(out)
        layer_2_out = self.layer2(layer_1_out)
        layer_3_out = self.layer3(layer_2_out)
        layer_4_out = self.layer4(layer_3_out)
        out = self.avgpool(layer_4_out)
        if self.shared_conv :
            out = out.reshape(batch_size, -1, out.shape[-3], out.shape[-2], out.shape[-1])
            out = self.globalpool2(out)
            out = self.bn1(out.squeeze(1))
            out = F.relu(out)

        out = torch.flatten(out, 1)
        out_abnormal = self.linear_abnormal(out)
        out_mtear = self.linear_mtear(out)
        out_acl = self.linear_acl(out)
        out_cartilage = self.linear_cartilage(out)

        if self.return_features:
            return [layer_1_out, layer_2_out, layer_3_out, layer_4_out]
        else:
            return [out_abnormal, out_mtear, out_acl, out_cartilage]


    def forward(self, kspace, data_space=None, shuffle=False):
        batch_size = kspace.shape[0]
        if data_space is not None:
            self.data_space = data_space

        if self.data_space == "ktoi_w_mag":
            out_mag = kspace.abs()
            out_mag = out_mag.to(torch.float32)
            out_mag = center_crop(out_mag, self.image_shape)
            out = self.conv1(out_mag)

        elif self.data_space == "ktoi_w_phase":
            out = kspace.angle()
            out = center_crop(out.to(torch.float32), self.image_shape)
            out = self.conv1(out)
        
        elif self.data_space == "ktoi_w_magphase":
            kspace = kspace.view(kspace.shape[0], kspace.shape[2], kspace.shape[3])
            angle = kspace.angle()
            mag = kspace.abs()

            out = torch.stack((mag, angle), axis=1).float()
            out = center_crop(out, self.image_shape)
            out = self.conv1_p(out)

        elif self.data_space == "ktoi_w_rss":
            out = kspace.to(torch.float32)
            out = self.conv1(out)
        
        out = self.get_outputs(out, batch_size)
        return out


def PreActResNet18(image_shape, data_space, input_channels=1, drop_prob=0.5, return_features=False, downsample=False, shared_conv=False, fc_layer_dim=100, norm='group'):
    return PreActResNet(
        PreActBlock, [2, 2, 2, 2], drop_prob=drop_prob, image_shape=image_shape, data_space=data_space, return_features=return_features, input_channels=input_channels, downsample=downsample, shared_conv=shared_conv, fc_layer_dim=fc_layer_dim, norm=norm
    )
