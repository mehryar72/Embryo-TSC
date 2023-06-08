import pretrainedmodels

from MYRESNO import BasicBlock, BasicBlockN, SEBasicBlock, BasicBlock1, conv1x1
from MYRESNO import Bottleneck, BottleneckN, SEBottleneck
# from modelZoo import *
from MYRESNO import _resnet, _resnet_MHA
# from BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6,BERTEmbedding2,BERTEmbedding
# from BERT.attention import MultiHeadedAttention, MultiHeadedAttention2
# from BERT.utils import SublayerConnection, PositionwiseFeedForward, SublayerConnection2
from TimeSeriesClass import *
from mymodel import MYTSC


# import cv2


class Multimodel(nn.Module):
    def __init__(self, model_name, n_cl):
        super(Multimodel, self).__init__()
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, n_cl)
        self.model = model

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        outputs = self.model(images)
        return outputs, outputs.view(BS[0], BS[1], -1), outputs


class mobileNet(nn.Module):
    def __init__(self, n_cl, prT=0):
        super(mobileNet, self).__init__()
        model = models.mobilenet_v2(pretrained=True if prT else False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(model.last_channel, n_cl),
        )
        self.model = model

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        outputs = self.model(images)
        return outputs, outputs.view(BS[0], BS[1], -1), outputs


class MultiLayer(nn.Module):
    def __init__(self, blck=1, n_cl=1, n_layers=3, layersL=[1, 1, 1], n_i_ch=16):
        super(MultiLayer, self).__init__()
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BasicBlock1
        layers = []
        self.inplanes = n_i_ch
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self._norm_layer = nn.BatchNorm2d
        layers.append(nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False))
        layers.append(nn.BatchNorm2d(self.inplanes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i in range(n_layers):
            layers = self._make_layer(block, n_i_ch * (2 ** (i)), layersL[i], layers)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.8)
        self.classifier = nn.Linear(self.inplanes, n_cl)

    def _make_layer(self, block, planes, blocks, layers, stride=1, dilate=False):
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

        # layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layers

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features = self.features(images).view(BS[0] * BS[1], -1)
        features = self.drop(features)
        outputs = self.classifier(features)
        return outputs, outputs.view(BS[0], BS[1], -1), features.view(BS[0], BS[1], -1)


def key_trans(old_key):
    return old_key.replace('resnet1.', '')

    return old_key


def rename_state_dict_keys(source, key_transformation=key_trans):
    state_dict = source
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict


class CMYRESAllp1pTL(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0, TL=0):
        super(CMYRESAllp1pTL, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        self.TL = TL

        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        # self.bn0 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        if TL == 0:
            self.linear0 = nn.Linear(resnet1.cout, n_cl * 2)
            self.act = nn.ReLU(inplace=True)
            self.linearA = nn.Linear(n_cl * 2 * 5, n_cl)
        else:
            self.linear0 = nn.Linear(resnet1.cout, hidden_size)
        # self.act0=nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear1 = nn.Linear(resnet1.cout, n_cl)
        # self.act1 = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear2 = nn.Linear(resnet1.cout, n_cl)
        # self.act2 = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear3 = nn.Linear(resnet1.cout, n_cl)
        # self.act3 = nn.ReLU(inplace=True)
        # self.bn4 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear4 = nn.Linear(resnet1.cout, n_cl)
        # self.act4 = nn.ReLU(inplace=True)
        # self.bnA = nn.BatchNorm1d(n_cl*5, momentum=0.001)

        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features11 = self.resnet1(images, BS[1])
        # features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        if not self.TL == 0:
            features11 = self.linear0(features11)
        features12 = features11.view(BS[0], BS[1], -1)
        # features12 = features12.reshape(features12.size(0), -1)
        # if self.drop:
        #     features2 = F.dropout(features11, p=0.5, training=self.training)
        # else:
        # features2=features11
        # features2=features2.view(BS[0],BS[1],-1)
        if self.TL == 0:

            out0 = self.linear0(features11 if not self.drop else F.dropout(features11, p=0.5, training=self.training))
            # out0=self.act(out0)
            out = out0.view(BS[0], -1)
            outputs = self.linearA(out)
        else:
            outputs = features11

        # out0=self.act0(out0)
        # # out0=self.bn0(out0)
        #
        # out1 = self.linear1(self.bn1(features2[:,1,:].squeeze()) if not self.drop else F.dropout(self.bn1(features2[:,1,:].squeeze()), p=0.5, training=self.training))
        # out1 = self.act1(out1)
        # # out1 = self.bn0(out1)
        #
        # out2 = self.linear2(self.bn2(features2[:,2,:].squeeze()) if not self.drop else F.dropout(self.bn2(features2[:,2,:].squeeze()), p=0.5, training=self.training))
        # out2 = self.act2(out2)
        # # out2 = self.bn0(out2)
        #
        # out3 = self.linear3(self.bn3(features2[:,3,:].squeeze()) if not self.drop else F.dropout(self.bn3(features2[:,3,:].squeeze()), p=0.5, training=self.training))
        # out3 = self.act3(out3)
        # # out3 = self.bn0(out3)
        #
        # out4 = self.linear4(self.bn4(features2[:,4,:].squeeze()) if not self.drop else F.dropout(self.bn4(features2[:,4,:].squeeze()), p=0.5, training=self.training))
        # out4 = self.act4(out4)
        # out4 = self.bn4(out4)

        # if self.drop:
        #     out0=F.dropout(out0, p=0.5, training=self.training)
        #     out1 = F.dropout(out1, p=0.5, training=self.training)
        #     out2 = F.dropout(out2, p=0.5, training=self.training)
        #     out3 = F.dropout(out3, p=0.5, training=self.training)
        #     out4 = F.dropout(out4, p=0.5, training=self.training)

        # outputs=self.linearA(torch.cat([out0,out1,out2],dim=1))

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        # outputs = self.linear(features2)
        return outputs, features11, features12


class CMYRESNOF(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1):
        super(CMYRESNOF, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
                         normalize_attn=normalize_attn)

        self.resnet = resnet
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.1)
        self.lstm = nn.LSTM(input_size=resnet.cout, hidden_size=hidden_size, num_layers=2, dropout=0.2,
                            bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.1)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(-optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.expand_as(images)
        features1 = self.resnet(images, BS[1])
        features1 = features1.reshape(features1.size(0), -1)
        features2, _ = self.lstm(features1.view(BS[0], BS[1], -1).permute(1, 0, 2))
        # features2=self.bn(features2[-1])
        features2 = F.relu(features2[-1])
        if self.drop:
            features2 = F.dropout(features2, p=0.5, training=self.training)

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        return outputs, features1, features2


class CMYRESALL32P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0, num_p=2, modelp=[]):
        super(CMYRESALL32P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resn = []
        resnet1 = _resnet('resnet18', block=block,
                          layers=[layers[0] % 10, layers[1] % 10, layers[2] % 10, layers[3] % 10],
                          pretrained=True if (prT % 10) else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        resn.append(resnet1)
        print(int(prT / 10) % 10)
        resnet2 = _resnet('resnet18', block=block,
                          layers=[int(layers[0] / 10) % 10, int(layers[1] / 10) % 10, int(layers[2] / 10) % 10,
                                  int(layers[3] / 10) % 10], pretrained=True if int(prT / 10) % 10 else False,
                          progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        resn.append(resnet2)
        if num_p == 3:
            resnet3 = _resnet('resnet18', block=block,
                              layers=[int(layers[0] / 100) % 10, int(layers[1] / 100) % 10, int(layers[2] / 100) % 10,
                                      int(layers[3] / 100) % 10], pretrained=True if int(prT / 100) % 10 else False,
                              progress=True, D=D,
                              Att=Att,
                              normalize_attn=normalize_attn)
            resn.append(resnet3)

        for ri, modelP in enumerate(modelp):
            pretrained_dict = torch.load(modelP, map_location=lambda storage, loc: storage)
            pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = resn[ri].state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            resn[ri].load_state_dict(model_dict)

        self.resnet1 = resn[0]
        self.resnet2 = resn[1]
        if num_p == 3:
            self.resnet3 = resn[2]
        self.bn0 = nn.BatchNorm1d(resnet1.cout * 3 if num_p == 3 else resnet1.cout * 2, momentum=0.001)
        self.actL = nn.ReLU(inplace=True)
        self.linear0 = nn.Linear(resnet1.cout * 3 if num_p == 3 else resnet1.cout * 2, n_cl)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        self.num_p = num_p
        # self.linear2 = nn.Linear(5, hidden_size)
        self.prT = prT

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        with torch.no_grad():
            if self.prT % 10:
                features11 = self.resnet1(images[:, 0, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
            else:
                features11 = self.resnet1(images[:, 0, 0, :, :].view(-1, 1, BS[3], BS[4]), BS[1])
            if int(self.prT / 10) % 10:
                features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
            else:
                features12 = self.resnet2(images[:, 1, 0, :, :].view(-1, 1, BS[3], BS[4]), BS[1])
            if self.num_p == 3:
                if int(self.prT / 100) % 10:
                    features13 = self.resnet2(images[:, 2, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
                else:
                    features13 = self.resnet2(images[:, 2, 0, :, :].view(-1, 1, BS[3], BS[4]), BS[1])
                features13 = features13.reshape(features12.size(0), -1)
            features11 = features11.reshape(features11.size(0), -1)
            features12 = features12.reshape(features12.size(0), -1)
        features2 = torch.cat([features11, features12, features13] if self.num_p == 3 else [features11, features12],
                              dim=1)
        if self.drop:
            features2 = F.dropout(features2, p=0.5, training=self.training)
        outputs = self.linear0(self.actL(features2))
        return outputs, features11, features2


class CMYRESALL322P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0, num_p=2, modelp=[]):
        super(CMYRESALL322P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resn = []
        resnet1 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                           layers=[layers[0] % 10, layers[1] % 10, layers[2] % 10, layers[3] % 10], Att=Att,
                           normalize_attn=normalize_attn,
                           hidden_size=hidden_size, optf=optf, prT=prT % 10)
        resn.append(resnet1)
        resnet2 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                           layers=[int(layers[0] / 10) % 10, int(layers[1] / 10) % 10, int(layers[2] / 10) % 10,
                                   int(layers[3] / 10) % 10], Att=Att, normalize_attn=normalize_attn,
                           hidden_size=hidden_size, optf=optf, prT=int(prT / 10) % 10)
        resn.append(resnet2)
        if num_p == 3:
            resnet3 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                               layers=[int(layers[0] / 100) % 10, int(layers[1] / 100) % 10, int(layers[2] / 100) % 10,
                                       int(layers[3] / 100) % 10], Att=Att, normalize_attn=normalize_attn,
                               hidden_size=hidden_size, optf=optf, prT=int(prT / 100) % 10)
            resn.append(resnet3)

        for ri, modelP in enumerate(modelp):
            pretrained_dict = torch.load(modelP, map_location=lambda storage, loc: storage)
            pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = resn[ri].state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            resn[ri].load_state_dict(model_dict)

        self.resnet1 = resn[0]
        self.resnet2 = resn[1]
        if num_p == 3:
            self.resnet3 = resn[2]
        self.bn0 = nn.BatchNorm1d(n_cl * 3 if num_p == 3 else n_cl * 2, momentum=0.001)
        self.linear0 = nn.Linear(n_cl * 3 if num_p == 3 else n_cl * 2, n_cl, bias=False)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        self.num_p = num_p
        # self.linear2 = nn.Linear(5, hidden_size)
        self.prT = prT

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        with torch.no_grad():
            if self.prT % 10:
                features11, _, _ = self.resnet1(images[:, 0, :, :, :], lens, optf)
            else:
                features11, _, _ = self.resnet1(images[:, 0, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
            if int(self.prT / 10) % 10:
                features12, _, _ = self.resnet2(images[:, 1, :, :, :], lens, optf)
            else:
                features12, _, _ = self.resnet2(images[:, 1, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
            if self.num_p == 3:
                if int(self.prT / 100) % 10:
                    features13 = self.resnet2(images[:, 2, :, :, :], lens, optf)
                else:
                    features13 = self.resnet2(images[:, 2, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
                features13 = features13.reshape(features12.size(0), -1)
            features11 = features11.reshape(features11.size(0), -1)
            features12 = features12.reshape(features12.size(0), -1)
        features2 = torch.cat([features11, features12, features13] if self.num_p == 3 else [features11, features12],
                              dim=1)
        if self.drop:
            features2 = F.dropout(features2, p=0.5, training=self.training)
        outputs = self.linear0(features2)
        return outputs, features11, features2


class CMYRESALL321P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0, num_p=2, modelp=[]):
        super(CMYRESALL321P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resn = []
        resnet1 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                           layers=[layers[0] % 10, layers[1] % 10, layers[2] % 10, layers[3] % 10], Att=Att,
                           normalize_attn=normalize_attn,
                           hidden_size=hidden_size, optf=optf, prT=prT % 10)
        resn.append(resnet1)
        resnet2 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                           layers=[int(layers[0] / 10) % 10, int(layers[1] / 10) % 10, int(layers[2] / 10) % 10,
                                   int(layers[3] / 10) % 10], Att=Att, normalize_attn=normalize_attn,
                           hidden_size=hidden_size, optf=optf, prT=int(prT / 10) % 10)
        resn.append(resnet2)
        if num_p == 3:
            resnet3 = CMYRES1P(FC=FC, n_cl=n_cl, D=D, drop=drop, GD=GD, blck=blck,
                               layers=[int(layers[0] / 100) % 10, int(layers[1] / 100) % 10, int(layers[2] / 100) % 10,
                                       int(layers[3] / 100) % 10], Att=Att, normalize_attn=normalize_attn,
                               hidden_size=hidden_size, optf=optf, prT=int(prT / 100) % 10)
            resn.append(resnet3)

        for ri, modelP in enumerate(modelp):
            pretrained_dict = torch.load(modelP, map_location=lambda storage, loc: storage)
            pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = resn[ri].state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # print(pretrained_dict)
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            resn[ri].load_state_dict(model_dict)

        self.resnet1 = resn[0]
        self.resnet2 = resn[1]
        if num_p == 3:
            self.resnet3 = resn[2]
        self.bn0 = nn.BatchNorm1d(n_cl * 3 if num_p == 3 else n_cl * 2, momentum=0.001)
        # self.linear0 = nn.Linear(n_cl*3 if num_p==3 else n_cl*2, n_cl,bias=False)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        self.num_p = num_p
        # self.linear2 = nn.Linear(5, hidden_size)
        self.prT = prT

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        with torch.no_grad():
            if self.prT % 10:
                features11, _, _ = self.resnet1(images[:, 0, :, :, :], lens, optf)
            else:
                features11, _, _ = self.resnet1(images[:, 0, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
            if int(self.prT / 10) % 10:
                features12, _, _ = self.resnet2(images[:, 1, :, :, :], lens, optf)
            else:
                features12, _, _ = self.resnet2(images[:, 1, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
            if self.num_p == 3:
                if int(self.prT / 100) % 10:
                    features13 = self.resnet2(images[:, 2, :, :, :], lens, optf)
                else:
                    features13 = self.resnet2(images[:, 2, 0, :, :].view(-1, 1, BS[3], BS[4]), lens, optf)
                features13 = features13.reshape(features12.size(0), -1)
            features11 = features11.reshape(features11.size(0), -1)
            features12 = features12.reshape(features12.size(0), -1)
        features2 = torch.cat([features11, features12, features13] if self.num_p == 3 else [features11, features12],
                              dim=1)
        # if self.drop:
        #     features2=F.dropout(features2, p=0.5, training=self.training)
        # outputs=self.linear0(features2)
        outputs = torch.mean(features2, dim=1, keepdim=True)
        return outputs, features11, features2


class CMYRES2P_CA(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, modelP=''):
        super(CMYRES2P_CA, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck

        # if prT:
        #     resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
        #                       normalize_attn=normalize_attn, prT=prT)
        #     resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
        #                       normalize_attn=normalize_attn, prT=prT)
        # else:
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        # res=torch.load(modelP)
        # resnet1=torch.load(modelP).resnet1
        # resnet2 = torch.load(modelP).resnet1
        if prT:
            pretrained_dict = torch.load(modelP, map_location=lambda storage, loc: storage)
            pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = resnet1.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dictss
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            resnet1.load_state_dict(model_dict)
            resnet2.load_state_dict(model_dict)
        self.resnet1 = resnet1
        self.resnet2 = resnet2
        self.bn = nn.BatchNorm1d(resnet1.cout + resnet2.cout, momentum=0.1)
        self.linear = nn.Linear(resnet1.cout + resnet2.cout, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)

        features11 = self.resnet1(images[:, 0, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        features12 = features12.reshape(features12.size(0), -1)
        if self.drop:
            features2 = F.dropout(torch.cat([features11, features12], dim=1), p=0.5, training=self.training)
        else:
            features2 = torch.cat([features11, features12], dim=1)
        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        return outputs, features11, features2


class CMYRES1P_CA(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, modelP=''):
        super(CMYRES1P_CA, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                         Att=Att,
                         normalize_attn=normalize_attn)
        if prT:
            pretrained_dict = torch.load(modelP, map_location=lambda storage, loc: storage)
            pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = resnet.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            resnet.load_state_dict(model_dict)

        # self.resnet = resnet

        # resnet = torch.load(modelP).resnet1
        # resnet2 = torch.load(modelP).resnet1
        self.resnet = resnet
        # self.resnet2 = resnet2
        self.bn = nn.BatchNorm1d(resnet.cout, momentum=0.1)
        self.linear = nn.Linear(resnet.cout, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(-optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.expand_as(images)
        features1 = self.resnet(images, BS[1])
        features1 = features1.reshape(features1.size(0), -1)
        if self.drop:
            features2 = F.dropout(features1, p=0.5, training=self.training)

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        return outputs, features1, features2


class CMYRES2P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRES2P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck

        # if prT:
        #     resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
        #                       normalize_attn=normalize_attn, prT=prT)
        #     resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
        #                       normalize_attn=normalize_attn, prT=prT)
        # else:
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        self.resnet2 = resnet2
        self.bn = nn.BatchNorm1d(resnet1.cout + resnet2.cout, momentum=0.1)
        self.linear = nn.Linear(resnet1.cout + resnet2.cout, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)

        features11 = self.resnet1(images[:, 0, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        features12 = features12.reshape(features12.size(0), -1)
        if self.drop:
            features2 = F.dropout(torch.cat([features11, features12], dim=1), p=0.5, training=self.training)
        else:
            features2 = torch.cat([features11, features12], dim=1)
        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        return outputs, features11, features2


class CMYRES1P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1):
        super(CMYRES1P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                         Att=Att,
                         normalize_attn=normalize_attn)

        self.resnet = resnet
        self.bn = nn.BatchNorm1d(resnet.cout, momentum=0.1)
        self.linear = nn.Linear(resnet.cout, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(-optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.expand_as(images)
        features1 = self.resnet(images, BS[1])
        features1 = features1.reshape(features1.size(0), -1)
        if self.drop:
            features2 = F.dropout(features1, p=0.5, training=self.training)

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        # features2=torch.mean(outputs.contiguous().view(BS[0],BS[1]),dim=1).unsqueeze(-1)
        return outputs, features1, features2


class CMYRES1P_MHA(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, lenght=96):
        super(CMYRES1P_MHA, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet = _resnet_MHA('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True,
                             D=D, lenght=lenght)

        self.resnet = resnet
        self.bn = nn.BatchNorm1d(resnet.cout, momentum=0.1)
        self.linear = nn.Linear(resnet.cout, n_cl)
        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(-optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.expand_as(images)
        features1 = self.resnet(images, BS[1])
        features1 = features1.reshape(features1.size(0), -1)
        if self.drop:
            features2 = F.dropout(features1, p=0.5, training=self.training)

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = self.linear(features2)
        return outputs, features1, features2


# class MYCNNTSC(nn.Module):
#     def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
#                  hidden_size=16, optf=1,prT=1,mod_path='',Frz=0,mode=0,D2=1,out=0,lenght=24,dilF=0,dil=1):
#         super(MYCNNTSC, self).__init__()
#         self.out=out
#         self.mode=mode
#         self.dilF=dilF
#         self.dil=dil
#         CNN=CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
#                  Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)
#         self.CNN = CNN
#         if prT:
#             pretrained_dict = torch.load(mod_path, map_location=lambda storage, loc: storage)
#             self.CNN.load_state_dict(pretrained_dict)
#             if Frz:
#
#                 for parameter in self.CNN.parameters():
#                     parameter.requires_grad = False
#                 self.CNN.eval()
#
#         self.Frz = Frz if prT else 0
#         if mode==0:
#             self.TSC=nn.LSTM(input_size=CNN.resnet.cout if out==0 else n_cl, hidden_size=hidden_size,
#                            bidirectional= False)
#             if hidden_size != n_cl:
#                 self.drop = nn.Dropout(p=0.7)
#                 # self.act = nn.ReLU(inplace=True)
#                 self.classifier = nn.Linear(in_features=hidden_size , out_features=n_cl)
#                 torch.nn.init.xavier_uniform_(self.classifier.weight)
#                 self.classifier.bias.data.zero_()
#
#         elif mode==1:
#             self.TSC = Classifier_FCN(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==102:
#             self.TSC = Classifier_FCN_Dil(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2,Dil=2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==104:
#             self.TSC = Classifier_FCN_Dil(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2,Dil=4)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==2:
#             self.TSC = Classifier_RESNET(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==3:
#
#             self.TSC = Classifier_FCN_FTA(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==310:
#
#             self.TSC = Classifier_FCN_FTA_H(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght,h=hidden_size)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==311:
#
#             self.TSC = Classifier_FCN_FTA_B_H(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght,h=hidden_size)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==301:
#
#             self.TSC = Classifier_FCN_FTA_B(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==8:
#
#             self.TSC = Classifier_FCN_FTA2(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==85:
#
#             self.TSC = Classifier_FCN_FTAMHA1(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==82:
#
#             self.TSC = Classifier_FCN_FTA22(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==83:
#
#             self.TSC = Classifier_FCN_FTA23(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==84:
#
#             self.TSC = Classifier_FCN_FTA24(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==4:
#
#             self.TSC = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,hidden_size=CNN.resnet.cout if out == 0 else n_cl,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==5:
#
#             self.TSC = Classifier_FCN_MHA3(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==7:
#
#             self.TSC = Classifier_FCN_MHA4(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==6:
#
#             self.TSC = Classifier_FCN_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==9:
#             self.TSC = Classifier_RESNET_MH1(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==10:
#             self.TSC = Classifier_RESNET_MH2(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==11:
#             self.TSC = Classifier_RESNET_MH3(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==12:
#             self.TSC = Classifier_RESNET_FTA1(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==13:
#             self.TSC = Classifier_RESNET_FTA2(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==14:
#
#
#             self.TSC = InceptionBlock(CNN.resnet.cout if out==0 else n_cl, n_filters=hidden_size, kernel_sizes=[9, 19, 39], bottleneck_channels=int(32/D2),
#                              use_residual=True, activation=nn.ReLU(), return_indices=False)
#             self.drop = nn.Dropout(p=0.7)
#             self.AVG=nn.AdaptiveAvgPool1d(1)
#             self.classifier = nn.Linear(in_features=hidden_size*4, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode == 15:
#
#             self.TSC = DilInceptionBlock(CNN.resnet.cout if out == 0 else n_cl, n_filters=hidden_size,
#                                       Dil_sizes=[2, 3, 4], bottleneck_channels=int(32 / D2),
#                                       use_residual=True, activation=nn.ReLU(), return_indices=False)
#             self.drop = nn.Dropout(p=0.7)
#             self.AVG = nn.AdaptiveAvgPool1d(1)
#             self.classifier = nn.Linear(in_features=hidden_size * 4, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#
#         if self.dilF==3:
#             self.Dilatt = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,hidden_size=CNN.resnet.cout if out == 0 else n_cl,length=dil)
#         elif self.dilF==4:
#             self.Dilatt = Classifier_FCN_MHA5(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=dil,D=D2)
#
#
#
#     def forward(self, images, lens, optf):
#         """Extract feature vectors from input images."""
#         BS = images.shape;
#         batchsize=BS[0]
#         images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
#
#         if not self.Frz:
#             outs,features,_ = self.CNN(images,lens,optf)
#         else:
#             self.CNN.eval()
#             with torch.no_grad():
#                 outs,features,_ = self.CNN(images,lens,optf)
#
#         if self.out:
#             features=outs.view(BS[0], BS[1], -1)
#         else:
#             features = features.view(BS[0], BS[1], -1)
#
#         if self.dilF==1 or self.dilF==2:
#             FS=features.shape
#             features=features.view(FS[0],-1,self.dil,FS[2]).transpose(1,2)
#             if self.training:
#                 features=features[:,random.randint(0,self.dil-1),:,:].squeeze(1)
#             else:
#                 features=features.contiguous().view(FS[0]*self.dil,-1,FS[2])
#                 batchsize=FS[0]*self.dil
#         elif self.dilF==3:
#             FS = features.shape
#             features = features.view(FS[0], -1, self.dil, FS[2])
#             features = features.contiguous().view(-1,self.dil, FS[2]).transpose(1,2)
#             features = self.Dilatt(features).contiguous().view(FS[0],-1, FS[2])
#         elif self.dilF==4:
#             FS = features.shape
#             features = features.view(FS[0], -1, self.dil, FS[2])
#             features = features.contiguous().view(-1,self.dil, FS[2]).transpose(1,2)
#             features = self.Dilatt(features).contiguous().view(FS[0],-1, FS[2])
#
#
#
#         if self.mode==0:
#             features = features.permute(1, 0, 2)
#             hiddens, (ht, ct) = self.TSC(features)
#             features2 = hiddens[-1]
#         elif self.mode==8 or self.mode==82 or self.mode==83 or self.mode==84 or self.mode==85:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
#         elif self.mode==13:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
#         elif self.mode==14 or self.mode==15:
#             features = features.permute(0, 2, 1)
#             features2 = self.AVG(self.TSC(features)).view(batchsize, -1)
#         else:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features).view(batchsize, -1)
#
#         features3=self.drop(features2)
#         outputs=self.classifier(features3)
#
#         if self.dilF==1:
#             if not self.training:
#                 outputs=torch.mean(outputs.view(FS[0],self.dil,-1),dim=1,keepdim=False)
#
#         elif self.dilF==2:
#             if not self.training:
#                 outputs=outputs.view(FS[0],self.dil,-1)[:,0,:]
#
#         elif self.dilF==0:
#             pass
#         else:
#             pass
#
#
#         return outputs, features2, torch.mean(outs.view(BS[0], BS[1], -1),dim=1,keepdim=False)

class CNNTSC(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, mod_path='', Frz=0, mode=0, D2=1, out=0, lenght=24, dilF=0, dil=1,
                 device='cuda', adaD=1, pos_enc=2):
        super(CNNTSC, self).__init__()
        self.out = out
        self.mode = mode
        self.dilF = dilF
        self.dil = dil
        CNN = CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
                       Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)
        self.CNN = CNN
        if prT:
            pretrained_dict = torch.load(mod_path, map_location=lambda storage, loc: storage)
            self.CNN.load_state_dict(pretrained_dict)
            if Frz:

                for parameter in self.CNN.parameters():
                    parameter.requires_grad = False
                self.CNN.eval()

        self.Frz = Frz if prT else 0
        self.TSC = MYTSC(mode=mode, input_shape=CNN.resnet.cout, hidden_size=hidden_size, n_cl=n_cl, lenght=lenght,
                         pow=2, LrEnb=1, LrMo=1, device=device, adaD=adaD, adaH=0, n_layers=1, pos_enc=pos_enc, ffh=4)
        # elif mode==14:
        #
        #
        #     self.TSC = InceptionBlock(CNN.resnet.cout if out==0 else n_cl, n_filters=hidden_size, kernel_sizes=[9, 19, 39], bottleneck_channels=int(32/D2),
        #                      use_residual=True, activation=nn.ReLU(), return_indices=False)
        #     self.drop = nn.Dropout(p=0.7)
        #     self.AVG=nn.AdaptiveAvgPool1d(1)
        #     self.classifier = nn.Linear(in_features=hidden_size*4, out_features=n_cl)
        #     torch.nn.init.xavier_uniform_(self.classifier.weight)
        #     self.classifier.bias.data.zero_()

        if self.dilF == 3:
            self.Dilatt = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,
                                          hidden_size=CNN.resnet.cout if out == 0 else n_cl, length=dil)
        elif self.dilF == 4:
            self.Dilatt = Classifier_FCN_MHA5(input_shape=CNN.resnet.cout if out == 0 else n_cl, length=dil, D=D2)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        batchsize = BS[0]
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])

        if not self.Frz:
            outs, features, _ = self.CNN(images, lens, optf)
        else:
            self.CNN.eval()
            with torch.no_grad():
                outs, features, _ = self.CNN(images, lens, optf)

        if self.out:
            features = outs.view(BS[0], BS[1], -1)
        else:
            features = features.view(BS[0], BS[1], -1)

        if self.dilF == 1 or self.dilF == 2:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2]).transpose(1, 2)
            if self.training:
                features = features[:, random.randint(0, self.dil - 1), :, :].squeeze(1)
            else:
                features = features.contiguous().view(FS[0] * self.dil, -1, FS[2])
                batchsize = FS[0] * self.dil
        elif self.dilF == 3:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2])
            features = features.contiguous().view(-1, self.dil, FS[2]).transpose(1, 2)
            features = self.Dilatt(features).contiguous().view(FS[0], -1, FS[2])
        elif self.dilF == 4:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2])
            features = features.contiguous().view(-1, self.dil, FS[2]).transpose(1, 2)
            features = self.Dilatt(features).contiguous().view(FS[0], -1, FS[2])

        # if self.mode==0:
        #     features = features.permute(1, 0, 2)
        #     hiddens, (ht, ct) = self.TSC(features)
        #     features2 = hiddens[-1]
        # elif self.mode==8 or self.mode==82 or self.mode==83 or self.mode==84 or self.mode==85:
        #     features = features.permute(0, 2, 1)
        #     features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
        # elif self.mode==13:
        #     features = features.permute(0, 2, 1)
        #     features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
        # elif self.mode==14 or self.mode==15:
        #     features = features.permute(0, 2, 1)
        #     features2 = self.AVG(self.TSC(features)).view(batchsize, -1)
        # else:
        #     features = features.permute(0, 2, 1)
        #     features2 = self.TSC(features).view(batchsize, -1)
        outputs, features2 = self.TSC(features)
        # features3=self.drop(features2)
        # outputs=self.classifier(features3)

        if self.dilF == 1:
            if not self.training:
                outputs = torch.mean(outputs.view(FS[0], self.dil, -1), dim=1, keepdim=False)

        elif self.dilF == 2:
            if not self.training:
                outputs = outputs.view(FS[0], self.dil, -1)[:, 0, :]

        elif self.dilF == 0:
            pass
        else:
            pass

        return outputs, features2, torch.mean(outs.view(BS[0], BS[1], -1), dim=1, keepdim=False)


class CNNTSC_CNN(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, mod_path='', Frz=0, mode=0, D2=1, out=0, lenght=24, dilF=0, dil=1,
                 device='cuda', adaD=1, pos_enc=2):
        super(CNNTSC_CNN, self).__init__()
        self.out = out
        self.mode = mode
        self.dilF = dilF
        self.dil = dil
        CNN = CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
                       Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)
        self.CNN = CNN
        if prT:
            pretrained_dict = torch.load(mod_path, map_location=lambda storage, loc: storage)
            self.CNN.load_state_dict(pretrained_dict)
            if Frz:

                for parameter in self.CNN.parameters():
                    parameter.requires_grad = False
                self.CNN.eval()

        self.Frz = Frz if prT else 0
        # self.TSC=MYT

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        batchsize = BS[0]
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])

        if not self.Frz:
            outs, features, _ = self.CNN(images, lens, optf)
        else:
            self.CNN.eval()
            with torch.no_grad():
                outs, features, _ = self.CNN(images, lens, optf)

        if self.out:
            features = outs.view(BS[0], BS[1], -1)
        else:
            features = features.view(BS[0], BS[1], -1)
        out3 = outs.view(BS[0], BS[1], -1)
        for ii in range(len(lens)):
            out3[ii, lens[ii]:, :] = 0
            print(lens[ii])
            print(out3.shape)

        return features, outs.view(BS[0], BS[1], -1), torch.mean(out3, dim=1, keepdim=False)


class CNNTSC_TSC(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, mod_path='', Frz=0, mode=0, D2=1, out=0, lenght=24, dilF=0, dil=1,
                 device='cuda', adaD=1, pos_enc=2, LrMo=2):
        super(CNNTSC_TSC, self).__init__()
        self.out = out
        self.mode = mode
        self.dilF = dilF
        self.dil = dil
        CNN = CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
                       Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)

        self.TSC = MYTSC(mode=mode, input_shape=CNN.resnet.cout, hidden_size=hidden_size, n_cl=n_cl, lenght=lenght,
                         pow=2, LrEnb=1, LrMo=LrMo, device=device, adaD=adaD, adaH=0, n_layers=1, pos_enc=pos_enc,
                         ffh=4)

        if self.dilF == 3:
            self.Dilatt = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,
                                          hidden_size=CNN.resnet.cout if out == 0 else n_cl, length=dil)
        elif self.dilF == 4:
            self.Dilatt = Classifier_FCN_MHA5(input_shape=CNN.resnet.cout if out == 0 else n_cl, length=dil, D=D2)

    def forward(self, features):
        """Extract feature vectors from input images."""

        if self.dilF == 1 or self.dilF == 2:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2]).transpose(1, 2)
            if self.training:
                features = features[:, random.randint(0, self.dil - 1), :, :].squeeze(1)
            else:
                features = features.contiguous().view(FS[0] * self.dil, -1, FS[2])
                batchsize = FS[0] * self.dil
        elif self.dilF == 3:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2])
            features = features.contiguous().view(-1, self.dil, FS[2]).transpose(1, 2)
            features = self.Dilatt(features).contiguous().view(FS[0], -1, FS[2])
        elif self.dilF == 4:
            FS = features.shape
            features = features.view(FS[0], -1, self.dil, FS[2])
            features = features.contiguous().view(-1, self.dil, FS[2]).transpose(1, 2)
            features = self.Dilatt(features).contiguous().view(FS[0], -1, FS[2])

        outputs, features2 = self.TSC(features)
        # features3=self.drop(features2)
        # outputs=self.classifier(features3)

        if self.dilF == 1:
            if not self.training:
                outputs = torch.mean(outputs.view(FS[0], self.dil, -1), dim=1, keepdim=False)

        elif self.dilF == 2:
            if not self.training:
                outputs = outputs.view(FS[0], self.dil, -1)[:, 0, :]

        elif self.dilF == 0:
            pass
        else:
            pass

        return outputs, features2


# class MYCNNTSC_e(nn.Module):
#     def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
#                  hidden_size=16, optf=1,prT=1,mod_path='',Frz=0,mode=0,D2=1,out=0,lenght=24,dilF=0,dil=1):
#         super(MYCNNTSC_e, self).__init__()
#         self.out=out
#         self.mode=mode
#         self.dilF=dilF
#         self.dil=dil
#         CNN=CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
#                  Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)
#         self.CNN = CNN
#         if prT:
#             pretrained_dict = torch.load(mod_path, map_location=lambda storage, loc: storage)
#             self.CNN.load_state_dict(pretrained_dict)
#             if Frz:
#
#                 for parameter in self.CNN.parameters():
#                     parameter.requires_grad = False
#                 self.CNN.eval()
#
#         self.Frz = Frz if prT else 0
#         if mode==0:
#             self.TSC=nn.LSTM(input_size=CNN.resnet.cout if out==0 else n_cl, hidden_size=hidden_size,
#                            bidirectional= False)
#             if hidden_size != n_cl:
#                 self.drop = nn.Dropout(p=0.7)
#                 # self.act = nn.ReLU(inplace=True)
#                 self.classifier = nn.Linear(in_features=hidden_size , out_features=n_cl)
#                 torch.nn.init.xavier_uniform_(self.classifier.weight)
#                 self.classifier.bias.data.zero_()
#
#         elif mode==1:
#             self.TSC = Classifier_FCN(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==102:
#             self.TSC = Classifier_FCN_Dil(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2,Dil=2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==104:
#             self.TSC = Classifier_FCN_Dil(input_shape=CNN.resnet.cout if out==0 else n_cl,D=D2,Dil=4)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==2:
#             self.TSC = Classifier_RESNET(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==3:
#
#             self.TSC = Classifier_FCN_FTA(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==8:
#
#             self.TSC = Classifier_FCN_FTA2(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==85:
#
#             self.TSC = Classifier_FCN_FTAMHA1(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==82:
#
#             self.TSC = Classifier_FCN_FTA22(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==83:
#
#             self.TSC = Classifier_FCN_FTA23(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==84:
#
#             self.TSC = Classifier_FCN_FTA24(input_shape=CNN.resnet.cout if out == 0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==4:
#
#             self.TSC = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,hidden_size=CNN.resnet.cout if out == 0 else n_cl,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==5:
#
#             self.TSC = Classifier_FCN_MHA3(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==7:
#
#             self.TSC = Classifier_FCN_MHA4(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode ==6:
#
#             self.TSC = Classifier_FCN_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=lenght,D=D2)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==9:
#             self.TSC = Classifier_RESNET_MH1(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==10:
#             self.TSC = Classifier_RESNET_MH2(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==11:
#             self.TSC = Classifier_RESNET_MH3(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==12:
#             self.TSC = Classifier_RESNET_FTA1(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==13:
#             self.TSC = Classifier_RESNET_FTA2(input_shape=CNN.resnet.cout if out==0 else n_cl, D=D2,length=lenght)
#             self.drop = nn.Dropout(p=0.7)
#             self.classifier = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode==14:
#
#
#             self.TSC = InceptionBlock(CNN.resnet.cout if out==0 else n_cl, n_filters=hidden_size, kernel_sizes=[9, 19, 39], bottleneck_channels=int(32/D2),
#                              use_residual=True, activation=nn.ReLU(), return_indices=False)
#             self.drop = nn.Dropout(p=0.7)
#             self.AVG=nn.AdaptiveAvgPool1d(1)
#             self.classifier = nn.Linear(in_features=hidden_size*4, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#         elif mode == 15:
#
#             self.TSC = DilInceptionBlock(CNN.resnet.cout if out == 0 else n_cl, n_filters=hidden_size,
#                                       Dil_sizes=[2, 3, 4], bottleneck_channels=int(32 / D2),
#                                       use_residual=True, activation=nn.ReLU(), return_indices=False)
#             self.drop = nn.Dropout(p=0.7)
#             self.AVG = nn.AdaptiveAvgPool1d(1)
#             self.classifier = nn.Linear(in_features=hidden_size * 4, out_features=n_cl)
#             torch.nn.init.xavier_uniform_(self.classifier.weight)
#             self.classifier.bias.data.zero_()
#
#         if self.dilF==3:
#             self.Dilatt = Classifier_BERT(input_shape=CNN.resnet.cout if out == 0 else n_cl,hidden_size=CNN.resnet.cout if out == 0 else n_cl,length=dil)
#         elif self.dilF==4:
#             self.Dilatt = Classifier_FCN_MHA5(input_shape=CNN.resnet.cout if out == 0 else n_cl,length=dil,D=D2)
#
#
#
#     def forward(self, images, lens, optf):
#         """Extract feature vectors from input images."""
#         BS = images.shape;
#         batchsize=BS[0]
#         images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
#
#         if not self.Frz:
#             outs,features,_ = self.CNN(images,lens,optf)
#         else:
#             self.CNN.eval()
#             with torch.no_grad():
#                 outs,features,_ = self.CNN(images,lens,optf)
#
#         if self.out:
#             features=outs.view(BS[0], BS[1], -1)
#         else:
#             features = features.view(BS[0], BS[1], -1)
#
#         if self.dilF==1 or self.dilF==2:
#             FS=features.shape
#             features=features.view(FS[0],-1,self.dil,FS[2]).transpose(1,2)
#             if self.training:
#                 features=features[:,random.randint(0,self.dil-1),:,:].squeeze(1)
#             else:
#                 features=features.contiguous().view(FS[0]*self.dil,-1,FS[2])
#                 batchsize=FS[0]*self.dil
#         elif self.dilF==3:
#             FS = features.shape
#             features = features.view(FS[0], -1, self.dil, FS[2])
#             features = features.contiguous().view(-1,self.dil, FS[2]).transpose(1,2)
#             features = self.Dilatt(features).contiguous().view(FS[0],-1, FS[2])
#         elif self.dilF==4:
#             FS = features.shape
#             features = features.view(FS[0], -1, self.dil, FS[2])
#             features = features.contiguous().view(-1,self.dil, FS[2]).transpose(1,2)
#             features = self.Dilatt(features).contiguous().view(FS[0],-1, FS[2])
#
#
#
#         if self.mode==0:
#             features = features.permute(1, 0, 2)
#             hiddens, (ht, ct) = self.TSC(features)
#             features2 = hiddens[-1]
#         elif self.mode==8 or self.mode==82 or self.mode==83 or self.mode==84 or self.mode==85:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
#         elif self.mode==13:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features,outs.view(batchsize, BS[1])).view(batchsize, -1)
#         elif self.mode==14 or self.mode==15:
#             features = features.permute(0, 2, 1)
#             features2 = self.AVG(self.TSC(features)).view(batchsize, -1)
#         else:
#             features = features.permute(0, 2, 1)
#             features2 = self.TSC(features).view(batchsize, -1)
#
#         features3=self.drop(features2)
#         outputs=self.classifier(features3)
#
#         if self.dilF==1:
#             if not self.training:
#                 outputs=torch.mean(outputs.view(FS[0],self.dil,-1),dim=1,keepdim=False)
#
#         elif self.dilF==2:
#             if not self.training:
#                 outputs=outputs.view(FS[0],self.dil,-1)[:,0,:]
#
#         elif self.dilF==0:
#             pass
#         else:
#             pass
#
#
#         return outputs, features2, outs.view(BS[0], BS[1])

class MYCNNTSC2(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=1, mod_path='', Frz=0, mode=0, D2=1, out=0, lenght=24):
        super(MYCNNTSC2, self).__init__()
        self.out = out
        self.mode = mode
        # CNN=CMYRES1P(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
        #          Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0)
        # self.CNN = CNN

        self.CNN2 = CMYRES1P_MHA(FC=1024, n_cl=1, D=D, blck=blck, layers=layers,
                                 Att=Att, normalize_attn=normalize_attn, hidden_size=hidden_size, optf=optf, prT=0,
                                 lenght=lenght)
        self.New = ['MH1', 'MH2', 'MH3', 'MH4', 'embeding']
        if prT:
            pretrained_dict = torch.load(mod_path, map_location=lambda storage, loc: storage)
            # self.CNN.load_state_dict(pretrained_dict)
            # pretrained_dict = rename_state_dict_keys(pretrained_dict)
            model_dict = self.CNN2.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.CNN2.load_state_dict(model_dict)
            if Frz:

                for parameter in self.CNN2.parameters():
                    parameter.requires_grad = False
                self.CNN2.eval()
                for name, child in self.CNN2.resnet.named_children():
                    if name in self.New:
                        # child.train(True)
                        for param in child.parameters():
                            param.requires_grad = True

        self.Frz = Frz if prT else 0
        self.classifier = nn.Linear(in_features=self.CNN2.resnet.cout, out_features=n_cl)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()
        self.drop = nn.Dropout(p=0.7)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])

        # self.CNN2.lenght=BS[1]
        if self.Frz:
            self.CNN2.eval()
            for name, child in self.CNN2.resnet.named_children():
                if name in self.New:
                    # child.train(True)
                    for param in child.parameters():
                        param.requires_grad = True

        outs, features, _ = self.CNN2(images, lens, optf)
        features3 = self.drop(features)
        outputs = self.classifier(features3)

        return outputs, features3, outs


class CMYRE2P(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRE2P, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        if prT:
            resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
            resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
            resnet3 = _resnet('resnet18', block=block, layers=layers, pretrained=True, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
        else:
            resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
            resnet2 = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
            resnet3 = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
                              normalize_attn=normalize_attn, prT=prT)
        # resnet4 = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
        #                   normalize_attn=normalize_attn)
        # resnet5 = _resnet('resnet18', block=block, layers=layers, pretrained=False, progress=True, D=D, Att=Att,
        #                   normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        self.resnet2 = resnet2
        self.resnet3 = resnet3
        # self.resnet4 = resnet4
        # self.resnet5 = resnet5

        # self.bn = nn.BatchNorm1d(resnet1.cout+resnet2.cout+resnet3.cout+resnet4.cout+resnet5.cout, momentum=0.001)
        # self.linear = nn.Linear(resnet1.cout+resnet2.cout+resnet3.cout+resnet4.cout+resnet5.cout, n_cl

        self.bn = nn.BatchNorm1d(resnet1.cout + resnet2.cout + resnet3.cout, momentum=0.001)
        self.linear = nn.Linear(resnet1.cout + resnet2.cout + resnet3.cout, hidden_size)

        self.optf = optf
        if self.optf:
            self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.1)
        self.drop = drop
        self.linear2 = nn.Linear(hidden_size, n_cl)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)

        features11 = self.resnet1(images[:, 0, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features13 = self.resnet3(images[:, 2, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        # features14 = self.resnet4(images[:, 3, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        # features15 = self.resnet5(images[:, 4, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        features12 = features12.reshape(features12.size(0), -1)
        features13 = features13.reshape(features13.size(0), -1)
        # features14 = features14.reshape(features14.size(0), -1)
        # features15 = features15.reshape(features15.size(0), -1)
        if self.drop:
            # features2 = F.dropout(torch.cat([features11,features12,features13,features14,features15],dim=1), p=0.5, training=self.training)
            features2 = F.dropout(torch.cat([features11, features12, features13], dim=1), p=0.5, training=self.training)

        else:
            # features2=torch.cat([features11,features12,features13,features14,features15],dim=1)
            features2 = torch.cat([features11, features12, features13], dim=1)

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        outputs = F.relu(self.linear(features2))
        if self.drop:
            outputs = F.dropout(outputs, p=0.5, training=self.training)
        outputs = self.linear2(outputs)
        return outputs, features11, features2


class CMYRESAllp(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRESAllp, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        self.bn0 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear0 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act0 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear1 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear2 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear3 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear4 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act4 = nn.ReLU(inplace=True)
        # self.bnA = nn.BatchNorm1d(n_cl*5, momentum=0.001)

        self.linearA = nn.Linear(n_cl * 2 * 5, n_cl)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features11 = self.resnet1(images, BS[1])
        # features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        # features12 = features12.reshape(features12.size(0), -1)
        # if self.drop:
        #     features2 = F.dropout(features11, p=0.5, training=self.training)
        # else:
        features2 = features11
        features2 = features2.view(BS[0], BS[1], -1)
        out0 = self.linear0(self.bn0(features2[:, 0, :].squeeze()) if not self.drop else F.dropout(
            self.bn0(features2[:, 0, :].squeeze()), p=0.5, training=self.training))
        out0 = self.act0(out0)
        # out0=self.bn0(out0)

        out1 = self.linear1(self.bn1(features2[:, 1, :].squeeze()) if not self.drop else F.dropout(
            self.bn1(features2[:, 1, :].squeeze()), p=0.5, training=self.training))
        out1 = self.act1(out1)
        # out1 = self.bn0(out1)

        out2 = self.linear2(self.bn2(features2[:, 2, :].squeeze()) if not self.drop else F.dropout(
            self.bn2(features2[:, 2, :].squeeze()), p=0.5, training=self.training))
        out2 = self.act2(out2)
        # out2 = self.bn0(out2)

        out3 = self.linear3(self.bn3(features2[:, 3, :].squeeze()) if not self.drop else F.dropout(
            self.bn3(features2[:, 3, :].squeeze()), p=0.5, training=self.training))
        # out3 = self.act3(out3)
        # out3 = self.bn0(out3)

        out4 = self.linear4(self.bn4(features2[:, 4, :].squeeze()) if not self.drop else F.dropout(
            self.bn4(features2[:, 4, :].squeeze()), p=0.5, training=self.training))
        out4 = self.act4(out4)
        # out4 = self.bn4(out4)

        if self.drop:
            outL = F.dropout(torch.cat([out0, out1, out2, out3, out4], dim=1), p=0.1, training=self.training)
            # out0 = F.dropout(out0, p=0.5, training=self.training)
            # out1 = F.dropout(out1, p=0.5, training=self.training)
            # out2 = F.dropout(out2, p=0.5, training=self.training)
            # out3 = F.dropout(out3, p=0.5, training=self.training)
            # out4 = F.dropout(out4, p=0.5, training=self.training)
        outputs = self.linearA(outL)
        # outputs=[out0, out1, out2, out3, out4]
        # outputs=self.linearA(torch.cat([out0,out1,out2],dim=1))

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        # outputs = self.linear(features2)
        return outputs, features11, features2


class CMYRESAllp13(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRESAllp13, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        self.bn0 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear0 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act0 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear1 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear2 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act2 = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear3 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act3 = nn.ReLU(inplace=True)
        # self.bn4 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear4 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act4 = nn.ReLU(inplace=True)
        # self.bnA = nn.BatchNorm1d(n_cl*5, momentum=0.001)

        self.linearA = nn.Linear(n_cl * 2 * 3, n_cl)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features11 = self.resnet1(images, BS[1])
        # features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        # features12 = features12.reshape(features12.size(0), -1)
        # if self.drop:
        #     features2 = F.dropout(features11, p=0.5, training=self.training)
        # else:
        features2 = features11
        features2 = features2.view(BS[0], BS[1], -1)
        out0 = self.linear0(self.bn0(features2[:, 0, :].squeeze()) if not self.drop else F.dropout(
            self.bn0(features2[:, 0, :].squeeze()), p=0.5, training=self.training))
        out0 = self.act0(out0)
        # out0=self.bn0(out0)

        out1 = self.linear1(self.bn1(features2[:, 1, :].squeeze()) if not self.drop else F.dropout(
            self.bn1(features2[:, 1, :].squeeze()), p=0.5, training=self.training))
        out1 = self.act1(out1)
        # out1 = self.bn0(out1)

        out2 = self.linear2(self.bn2(features2[:, 2, :].squeeze()) if not self.drop else F.dropout(
            self.bn2(features2[:, 2, :].squeeze()), p=0.5, training=self.training))
        out2 = self.act2(out2)
        # out2 = self.bn0(out2)

        # out3 = self.linear3(self.bn3(features2[:, 3, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn3(features2[:, 3, :].squeeze()), p=0.5, training=self.training))
        # out3 = self.act3(out3)
        # out3 = self.bn0(out3)

        # out4 = self.linear4(self.bn4(features2[:, 4, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn4(features2[:, 4, :].squeeze()), p=0.5, training=self.training))
        # out4 = self.act4(out4)
        # out4 = self.bn4(out4)

        if self.drop:
            outL = F.dropout(torch.cat([out0, out1, out2], dim=1), p=0.1, training=self.training)
            # out0 = F.dropout(out0, p=0.5, training=self.training)
            # out1 = F.dropout(out1, p=0.5, training=self.training)
            # out2 = F.dropout(out2, p=0.5, training=self.training)
            # out3 = F.dropout(out3, p=0.5, training=self.training)
            # out4 = F.dropout(out4, p=0.5, training=self.training)
        outputs = self.linearA(outL)
        # outputs=[out0, out1, out2, out3, out4]
        # outputs=self.linearA(torch.cat([out0,out1,out2],dim=1))

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        # outputs = self.linear(features2)
        return outputs, features11, features2


class CMYRESAllp45(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRESAllp45, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        # self.bn0 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear0 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act0 = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear1 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act1 = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear2 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear3 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear4 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act4 = nn.ReLU(inplace=True)
        # self.bnA = nn.BatchNorm1d(n_cl*5, momentum=0.001)

        self.linearA = nn.Linear(n_cl * 2 * 2, n_cl)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features11 = self.resnet1(images, BS[1])
        # features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        # features12 = features12.reshape(features12.size(0), -1)
        # if self.drop:
        #     features2 = F.dropout(features11, p=0.5, training=self.training)
        # else:
        features2 = features11
        features2 = features2.view(BS[0], BS[1], -1)
        # out0 = self.linear0(self.bn0(features2[:, 0, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn0(features2[:, 0, :].squeeze()), p=0.5, training=self.training))
        # out0 = self.act0(out0)
        # out0=self.bn0(out0)

        # out1 = self.linear1(self.bn1(features2[:, 1, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn1(features2[:, 1, :].squeeze()), p=0.5, training=self.training))
        # out1 = self.act1(out1)
        # out1 = self.bn0(out1)

        # out2 = self.linear2(self.bn2(features2[:, 2, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn2(features2[:, 2, :].squeeze()), p=0.5, training=self.training))
        # out2 = self.act2(out2)
        # out2 = self.bn0(out2)

        out3 = self.linear3(self.bn3(features2[:, 3, :].squeeze()) if not self.drop else F.dropout(
            self.bn3(features2[:, 3, :].squeeze()), p=0.5, training=self.training))
        # out3 = self.act3(out3)
        # out3 = self.bn0(out3)

        out4 = self.linear4(self.bn4(features2[:, 4, :].squeeze()) if not self.drop else F.dropout(
            self.bn4(features2[:, 4, :].squeeze()), p=0.5, training=self.training))
        out4 = self.act4(out4)
        # out4 = self.bn4(out4)

        if self.drop:
            outL = F.dropout(torch.cat([out3, out4], dim=1), p=0.1, training=self.training)
            # out0 = F.dropout(out0, p=0.5, training=self.training)
            # out1 = F.dropout(out1, p=0.5, training=self.training)
            # out2 = F.dropout(out2, p=0.5, training=self.training)
            # out3 = F.dropout(out3, p=0.5, training=self.training)
            # out4 = F.dropout(out4, p=0.5, training=self.training)
        outputs = self.linearA(outL)
        # outputs=[out0, out1, out2, out3, out4]
        # outputs=self.linearA(torch.cat([out0,out1,out2],dim=1))

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        # outputs = self.linear(features2)
        return outputs, features11, features2


class CMYRESAllp5(nn.Module):
    def __init__(self, FC=1024, n_cl=2, D=2, drop=1, GD=1, blck=1, layers=[1, 1, 1, 1], Att=3, normalize_attn=False,
                 hidden_size=16, optf=1, prT=0):
        super(CMYRESAllp5, self).__init__()
        # block=BasicBlock if blck==1 else Bottleneck
        if blck == 0:
            block = BasicBlockN
        elif blck == 1:
            block = BasicBlock
        elif blck == 2:
            block = BottleneckN
        elif blck == 3:
            block = Bottleneck
        elif blck == 4:
            block = SEBasicBlock
        elif blck == 5:
            block = SEBottleneck
        resnet1 = _resnet('resnet18', block=block, layers=layers, pretrained=True if prT else False, progress=True, D=D,
                          Att=Att,
                          normalize_attn=normalize_attn)
        self.resnet1 = resnet1
        # self.bn0 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear0 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act0 = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear1 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act1 = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear2 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act2 = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        # self.linear3 = nn.Linear(resnet1.cout, n_cl*2)
        # self.act3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm1d(resnet1.cout, momentum=0.001)
        self.linear4 = nn.Linear(resnet1.cout, n_cl * 2)
        self.act4 = nn.ReLU(inplace=True)
        # self.bnA = nn.BatchNorm1d(n_cl*5, momentum=0.001)

        self.linearA = nn.Linear(n_cl * 2 * 1, n_cl)
        self.optf = optf
        if self.optf:
            if prT:
                self.conv1 = nn.Conv2d(3, 3, bias=False, kernel_size=7, stride=1, padding=3)
            else:
                self.conv1 = nn.Conv2d(1, 1, bias=False, kernel_size=7, stride=1, padding=3)
            self.bn1 = nn.BatchNorm2d(1, momentum=0.001)
        self.drop = drop
        # self.linear2 = nn.Linear(5, hidden_size)

    def forward(self, images, lens, optf):
        """Extract feature vectors from input images."""
        BS = images.shape;
        # images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        # optf = optf.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.optf:
            optf = torch.mean(optf, dim=1) + 2
            M = self.conv1(optf)
            # M = self.bn1(M)
            M = F.sigmoid(M)
            images = images * M.view(-1, 1, BS[2], BS[3], BS[4]).expand_as(images)
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        features11 = self.resnet1(images, BS[1])
        # features12 = self.resnet2(images[:, 1, :, :, :].view(-1, BS[2], BS[3], BS[4]), BS[1])
        features11 = features11.reshape(features11.size(0), -1)
        # features12 = features12.reshape(features12.size(0), -1)
        # if self.drop:
        #     features2 = F.dropout(features11, p=0.5, training=self.training)
        # else:
        features2 = features11
        features2 = features2.view(BS[0], BS[1], -1)
        # out0 = self.linear0(self.bn0(features2[:, 0, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn0(features2[:, 0, :].squeeze()), p=0.5, training=self.training))
        # out0 = self.act0(out0)
        # out0=self.bn0(out0)

        # out1 = self.linear1(self.bn1(features2[:, 1, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn1(features2[:, 1, :].squeeze()), p=0.5, training=self.training))
        # out1 = self.act1(out1)
        # out1 = self.bn0(out1)

        # out2 = self.linear2(self.bn2(features2[:, 2, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn2(features2[:, 2, :].squeeze()), p=0.5, training=self.training))
        # out2 = self.act2(out2)
        # out2 = self.bn0(out2)

        # out3 = self.linear3(self.bn3(features2[:, 3, :].squeeze()) if not self.drop else F.dropout(
        #    self.bn3(features2[:, 3, :].squeeze()), p=0.5, training=self.training))
        # out3 = self.act3(out3)
        # out3 = self.bn0(out3)

        out4 = self.linear4(self.bn4(features2[:, 4, :].squeeze()) if not self.drop else F.dropout(
            self.bn4(features2[:, 4, :].squeeze()), p=0.5, training=self.training))
        out4 = self.act4(out4)
        # out4 = self.bn4(out4)

        # if self.drop:
        #    outL = F.dropout(torch.cat([out3, out4], dim=1), p=0.1, training=self.training)
        # out0 = F.dropout(out0, p=0.5, training=self.training)
        # out1 = F.dropout(out1, p=0.5, training=self.training)
        # out2 = F.dropout(out2, p=0.5, training=self.training)
        # out3 = F.dropout(out3, p=0.5, training=self.training)
        # out4 = F.dropout(out4, p=0.5, training=self.training)
        outputs = self.linearA(out4)
        # outputs=[out0, out1, out2, out3, out4]
        # outputs=self.linearA(torch.cat([out0,out1,out2],dim=1))

        # features2 = F.relu(self.linear(features1))
        # if self.drop:
        #     features2 = F.dropout(features2, p=0.5, training=self.training)
        # outputsL = self.linear2(lens)
        # outputsL=self.bn2(outputsL)
        # if self.drop:
        #     outputsL=F.dropout(outputsL, p=0.5, training=self.training)
        # outputs = self.linear(torch.cat([features2, outputsL], dim=1))
        # outputs = self.linear(features2)
        return outputs, features11, features2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
