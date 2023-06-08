# from .utils import load_state_dict_from_url
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BERT.attention import MultiHeadedAttention3
from BERT.bert import BERTEmbedding_RE
from BERT.utils import SublayerConnection

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': "/home/mabbasib/resnet18-5c106cde.pth",
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def flowC(imgs):
    ish = imgs.shape
    prvs = imgs[0, :, :, :].squeeze()
    for i in range(1, ish[0]):
        next = imgs[i, :, :, :].squeeze()
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=1, winsize=1, iterations=1, poly_n=5,
                                            poly_sigma=1.2, flags=0)
        mag1, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=1, winsize=1, iterations=1, poly_n=5,
        #                                     poly_sigma=20, flags=0)
        # mag2, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mag2=mag1
        prvs = next
        # temp=np.expand_dims(np.stack([mag1,mag2],axis=0),axis=0)
        temp = np.expand_dims(np.expand_dims(mag1, axis=0), axis=0)
        if i == 1:
            masks = temp
        masks = np.concatenate([masks, temp], axis=0)
    return masks


class FlowAttention(nn.Module):
    def __init__(self, in_features, normalize_attn=0, outp=1):
        super(FlowAttention, self).__init__()
        self.normalize_attn = normalize_attn
        self.op1 = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
        # self.op2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, bias=False)
        # self.op2 = nn.Upsample(size=112, mode='bilinear')

    def forward(self, input, SL):
        N, C, W, H = input.size()
        device = input.device
        x = self.op1(input)
        # x = self.op2(x)
        # x=F.relu(x)
        # y=x.view(-1,SL,1,W,H)
        y2 = torch.tensor(x.view(-1, SL, 1, W, H), requires_grad=False)
        npx = np.array(y2.clone().detach().cpu())
        npmasks = (lambda imgs: np.stack([flowC(img) for img in imgs]))(list(npx))
        x.data = torch.tensor(npmasks, requires_grad=False).to(device).view(N, 1, W, H).data
        if self.normalize_attn == 0:
            # a = F.softmax(x.view(N,1,-1), dim=2).view(N,1,W,H)
            a = torch.sigmoid(x) - 0.5
        else:
            a = torch.sigmoid(x)
        # g = torch.mul(a.expand_as(input) * 2, input)
        g = torch.mul(a.expand_as(input) * 2, input)
        # g = g * torch.max(input)/torch.max(g)
        return a.view(N, 1, W, H), g


class LinearAttentionBlock1(nn.Module):
    def __init__(self, in_features, normalize_attn=True, outp=1):
        super(LinearAttentionBlock1, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, SL):
        N, C, W, H = l.size()
        c = self.op(l)  # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        # if self.normalize_attn:
        #     g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        # else:
        #     g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return a.view(N, 1, W, H), g


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
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


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=0.01)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckN(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, pretrained, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, D=1, Att=0, normalize_attn=True):
        super(ResNet, self).__init__()
        if Att == 1:
            self.att = 1
            atblock = LinearAttentionBlock1
        elif Att == 3:
            self.att = 3
            atblock = FlowAttention
        else:
            self.att = 0

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 / D)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if pretrained:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 / D), layers[0])

        self.layer2 = self._make_layer(block, int(128 / D), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, int(256 / D), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, int(512 / D), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cout = self.inplanes
        if not self.att == 0:
            self.A12 = atblock(in_features=int(64 / D), normalize_attn=normalize_attn)
            self.A1 = atblock(in_features=int(64 * block.expansion / D), normalize_attn=normalize_attn)
            self.A2 = atblock(in_features=int(128 * block.expansion / D), normalize_attn=normalize_attn)
            self.A3 = atblock(in_features=int(256 * block.expansion / D), normalize_attn=normalize_attn)
            # self.A4 = atblock(in_features=int(512*block.expansion / D), normalize_attn=normalize_attn,outp=1)
            # self.A5 = atblock(in_features=int(512 * block.expansion / D), normalize_attn=normalize_attn, outp=1)
            # self.A5 = atblock(in_features=int(512 / D), normalize_attn=normalize_attn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, SL):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.att == 0:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.att == 1:
            _, x = self.A12(x, SL)
            x = self.layer1(x)
            _, x = self.A1(x, SL)
            x = self.layer2(x)
            _, x = self.A2(x, SL)
            x = self.layer3(x)
            _, x = self.A3(x, SL)
            x = self.layer4(x)
            # _, x = self.A4(x,SL)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x, SL):
        return self._forward_impl(x, SL)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, pretrained, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class MHA(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention3(h=attn_heads, d_model=hidden)
        # self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x = x.permute(0, 2, 1)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class ResNet_MHA(nn.Module):

    def __init__(self, block, layers, pretrained, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, D=1, lenght=96):
        super(ResNet_MHA, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 / D)
        self.dilation = 1
        self.attheads = 1
        self.drop = 0.5
        self.mask_prob = 0.5
        self.lenght = lenght
        self.OlD_length = lenght

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if pretrained:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 / D), layers[0])
        # self.embeding = BERTEmbedding2_RE(self.inplanes,lenght)
        self.embeding = BERTEmbedding_RE(self.inplanes, lenght)

        self.MH1 = MHA(self.inplanes, 1, self.drop)
        self.maxpool1D1 = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        self.layer2 = self._make_layer(block, int(128 / D), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.MH2 = MHA(self.inplanes, 2, self.drop)
        self.maxpool1D2 = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        self.layer3 = self._make_layer(block, int(256 / D), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.MH3 = MHA(self.inplanes, 4, self.drop)
        self.maxpool1D3 = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        self.layer4 = self._make_layer(block, int(512 / D), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.MH4 = MHA(self.inplanes, 8, self.drop)
        self.maxpool1D4 = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cout = self.inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, SL):
        # See note [TorchScript super()]

        self.OlD_length = self.lenght
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        XS = x.shape
        x = self.embeding(x.view(-1, self.lenght, x.shape[-3], x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2))
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, XS[1], XS[2], XS[3])
        x = self._Temporal(x, self.MH1, self.maxpool1D1)
        x = self.layer2(x)
        x = self._Temporal(x, self.MH2, self.maxpool1D2)
        x = self.layer3(x)
        x = self._Temporal(x, self.MH3, self.maxpool1D3)
        x = self.layer4(x)
        x = self._Temporal(x, self.MH4, self.maxpool1D4)

        x = self.avgpool(x.view(-1, self.lenght, x.shape[-3], x.shape[-2], x.shape[-1]).transpose(1, 2))
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        self.lenght = self.OlD_length

        return x

    def forward(self, x, SL):
        return self._forward_impl(x, SL)

    def _Cmask(self, batch_size):
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.lenght).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, 1, 1, self.lenght, self.lenght).cuda()
        return mask

    def _Temporal(self, x, MH, MP):
        XS = x.shape
        x = x.view(-1, self.lenght, XS[1], XS[2], XS[3]).permute(0, 3, 4, 1, 2)
        x = MH(x, self._Cmask(x.shape[0]))
        x = MP(x)
        self.lenght = x.shape[-2]
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, XS[1], XS[2], XS[3])
        return x


def _resnet_MHA(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_MHA(block, layers, pretrained, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
