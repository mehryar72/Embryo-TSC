# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:27:35 2019

@author: mabbasib
"""


import argparse

import logging
import sys

import torchvision.transforms as transforms
from sklearn.metrics import jaccard_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from DataLoad import *
from modelsN import *



parser = argparse.ArgumentParser()
parser.add_argument("--SL", type=int, default=1)
parser.add_argument("--tt", type=int, default=62)

parser.add_argument("--BS", type=int, default=4096)
parser.add_argument("--Sc", type=int, default=0)
parser.add_argument("--scSt", type=int, default=0)
parser.add_argument("--An", type=int, default=1)
parser.add_argument("--dd", type=int, default=1)
parser.add_argument("--JobId", type=int, default=11)

parser.add_argument("--NN", type=int, default=41)

parser.add_argument("--crp", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--lrW", type=int, default=2)
parser.add_argument("--L1", type=int, default=1)
parser.add_argument("--L2", type=int, default=1)
parser.add_argument("--L3", type=int, default=2)
parser.add_argument("--L4", type=int, default=2)
parser.add_argument("--D", type=int, default=4)
parser.add_argument("--nlayer", type=int, default=3)
parser.add_argument("--Ra", type=int, default=3)
parser.add_argument("--blck", type=int, default=1)
parser.add_argument("--Att", type=int, default=0)
parser.add_argument("--opt", type=int, default=0)
parser.add_argument("--ims", type=int, default=224)
parser.add_argument("--ScR", type=int, default=0)
parser.add_argument("--hidS", type=int, default=128)
parser.add_argument("--optf", type=int, default=0)
parser.add_argument("--NoD", type=int, default=1)
parser.add_argument("--ccut", type=int, default=1)
parser.add_argument("--Dil", type=int, default=1)

parser.add_argument("--epochR", type=int, default=5)
parser.add_argument("--EnbB", type=int, default=0)
parser.add_argument("--prT", type=int, default=1)
parser.add_argument("--nch", type=int, default=8)


args = parser.parse_args()

dd = args.dd


def save_models(epoch):
    torch.save(net1.state_dict(), netS)


def reslog(line):
    with open(r'./acc_eval_n.csv', 'a', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(line)
        f.close()


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.shutdown()
if args.Sc:
    iii = 1
else:
    iii = 0

name = './rundataDay3_r/train_day' + (''.join(sys.argv[1:]))
# logF = name + '.csv'
netS = name + '.pth'
out_name = './rundataE/outDay3-{}-{}-{}_'.format(args.Ra, args.lr, args.JobId)
save_path = './EmbyroFeatures/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# root_dir = "./home/mabbasib/Data_c2_S1R/t (6)/"
opt_path = "./Data_c2_S1R_opf2/"
length_paths = "./out.csv"
Cord_path = "dictCord.pkl"
root_dir = './D_ac/'

day_index_path = './ix48.csv'


seq_length = args.SL
NW = 0

batchS = args.BS
ratio = [0.85, 0.15, 0.0]
best_acc = 0.0
Sc = args.Sc
num_epochs = 100 if args.dd else 100


if args.NN == 9:
    net1 = CMYRES1P(FC=1024, n_cl=1, D=args.D, blck=args.blck, layers=[args.L1, args.L2, args.L3, args.L4],
                    Att=args.Att, normalize_attn=args.ScR, hidden_size=args.hidS, optf=args.optf, prT=args.prT)

print(count_parameters(net1))

print(list(net1.children()))
net1 = net1.to(device)
print(10 ** -args.lrW)
loss_fn = torch.nn.BCEWithLogitsLoss()
c_p1, c_p2, e_i1, e_i2 = datPP3(root_dir, args.Ra)
if args.prT:
    p = 3
else:
    p = 1
transform = transforms.Compose([

    # transforms.RandomRotation(args.An * 45, resample=Image.BICUBIC),
    transforms.RandomAffine(args.An * 45, translate=(0.05, 0.05), scale=None, shear=2, resample=Image.NEAREST,
                            fillcolor=102 if args.crp else 0),
    transforms.ColorJitter(brightness=0.5, contrast=0.05),
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
    # Crp(c=args.crp),
    # transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(args.ims, scale=(0.85, 1.05), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    # RepC(p=1),
    transforms.Normalize([0.4], [0.2]),
    RepC(p=p)
])

transformV = transforms.Compose([
    # Crp(c=args.crp),
    transforms.Resize((args.ims, args.ims)),
    transforms.ToTensor(),
    # RepC(p=1),
    transforms.Normalize([0.4],
                         [0.2]),
    RepC(p=p),

])

epochR = args.epochR


samTr = MySamplerDilate(e_i1, args.Dil)
samVa = MySamplerDilate(e_i2, 1)


if not args.dd:
    TrainD = MyDatasetDivide1p_Day(
        image_paths=c_p1,
        seq_length=seq_length,
        transform=transform,
        length=len(samTr),
        end_idx=e_i1,
        C_path=Cord_path,
        L_path=length_paths,
        O_path=opt_path,
        st_path=day_index_path,
        crpc=args.crp, opt=args.optf, NoD=args.NoD, cut=args.ccut)
else:
    TrainD = MyDatasetDivideNrep_Day_order(
        image_paths=c_p1,
        seq_length=seq_length,
        transform=transformV,
        length=len(samTr),
        end_idx=e_i1,
        C_path=Cord_path,
        L_path=length_paths,
        O_path=opt_path,
        st_path=day_index_path,
        crpc=args.crp, opt=args.optf, NoD=args.NoD, cut=args.ccut)
LdT = DataLoader(
    TrainD,
    batch_size=batchS,
    sampler=samTr,
    num_workers=NW,
    pin_memory=False
)

ValiD = MyDatasetDivideNrep_Day_order(
    image_paths=c_p2,
    seq_length=seq_length,
    transform=transformV,
    # length=len(e_i2)-1,
    length=len(samVa),
    end_idx=e_i2,
    C_path=Cord_path,
    L_path=length_paths,
    st_path=day_index_path,
    O_path=opt_path, crpc=args.crp, opt=args.optf, NoD=args.NoD, cut=args.ccut)

LdV = DataLoader(
    ValiD,
    batch_size=batchS,
    sampler=samVa,
    num_workers=NW,
    pin_memory=False
)

#########################################################################

data_loaders = {"train": LdT, "val": LdV}
data_lengths = {"train": samTr, "val": samVa}
tloss = 0
vloss = 0
tac = 0
vac = 0
lgloss = {"train": tloss, "val": vloss}
lgacc = {"train": tac, "val": vac}
best_acc = 0
be_loss_v = 0
be_loss_tr = 0
cc = 0
inv = 0
bep = 0
epochi = 1
tr_bb = 0

data_loaders = {"train": LdT, "val": LdV}
data_lengths = {"train": len(TrainD), "val": len(ValiD)}
e_i = {"train": e_i1, "val": e_i2}
tloss = 0
vloss = 0
tac = 0
vac = 0
lgloss = {"train": tloss, "val": vloss}
lgacc = {"train": tac, "val": vac}

cc = 0
inv = 0

pretrained_dict = torch.load(netS, map_location=lambda storage, loc: storage)
net1.load_state_dict(pretrained_dict)
net1 = net1.to(device)
for epoch in range(1):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['val', 'train']:

        net1.train(False)  # Set model to evaluate mode
        epoch_loss = [None] * 1
        t_a = [None] * 1
        seq_acc = [None] * 2
        out = [[] for x in range(len(e_i[phase]) - 1)]
        feat = [[] for x in range(len(e_i[phase]) - 1)]
        IDL = [None] * (len(e_i[phase]) - 1)
        LenL = [None] * (len(e_i[phase]) - 1)
        labelsA = torch.zeros(len(e_i[phase]) - 1)
        outs = torch.zeros(len(e_i[phase]) - 1)
        outs2 = torch.zeros(len(e_i[phase]) - 1)
        Scorelist = []
        preA = [None] * 1
        recA = [None] * 1
        fsA = [None] * 1
        J_sA = [None] * 1
        preS = [None] * 2
        recS = [None] * 2
        fsS = [None] * 2
        J_sS = [None] * 2
        aucA = [None] * 1
        aucS = [None] * 2

        running_loss = 0.0
        train_acc = 0
        t_a_e = 0
        for data in data_loaders[phase]:
            if args.NN == 9:
                images = data[0][:, 0, :, :, :]
                labels = data[1]
                lens = data[3]
                optf = data[4]
            else:
                images = data[0]
                labels = data[1]
                lens = data[3]
                optf = data[4]
            index = data[2]
            images = images.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs, global_feat, _ = net1(images, lens, optf)
                for i in range(len(outputs)):
                    out[index[i]].append(outputs[i].cpu())
                    feat[index[i]].append(global_feat[i].cpu())

                    IDL[index[i]] = data[5][i]
                    labelsA[index[i]] = labels[i].cpu()
                loss = loss_fn(outputs, labels.float())
                Scorelist.append(torch.cat([outputs.cpu(), labels.float().cpu()], dim=1))
            running_loss += loss.data.item() * float(len(data[1]))
            pred = outputs.data.clone()
            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            prediction = pred
            train_acc += torch.sum(prediction == labels.float())
            print(float(torch.sum(prediction == labels.float())) / float(len(data[1])))
        epoch_loss[0] = float(running_loss) / float(data_lengths[phase])

        t_a[0] = float(train_acc) / float(data_lengths[phase])
        for i in range(len(out)):
            outs[i] = torch.mean(torch.tensor(out[i]), dim=0)
            outs2[i] = torch.mean(torch.sigmoid(torch.tensor(out[i])).round(), dim=0)
            LenL[i] = len(feat[i])
            if len(feat[i]) >= 96:
                feat[i] = np.vstack(feat[i][:96])
                LenL[i] = 96
            else:
                feat[i] = np.pad(np.vstack(feat[i]), ((0, 96 - len(feat[i])), (0, 0)))
        pred = outs.data.clone()
        pred[pred >= 0] = 1
        pred[pred < 0] = 0
        prediction = pred
        seq_acc[0] = float(torch.sum(prediction == labelsA.float().cpu())) / len(labelsA)
        Scorearr = torch.cat(Scorelist)
        preA[0], recA[0], fsA[0], _ = precision_recall_fscore_support(np.array(Scorearr[:, 1]),
                                                                      np.array(torch.sigmoid(Scorearr[:, 0])).round(),
                                                                      average='macro')
        preS[0], recS[0], fsS[0], _ = precision_recall_fscore_support(np.array(labelsA.float().cpu()),
                                                                      np.array(torch.sigmoid(outs)).round(),
                                                                      average='macro')
        preS[1], recS[1], fsS[1], _ = precision_recall_fscore_support(np.array(labelsA.float().cpu()),
                                                                      np.array(outs2).round(), average='macro')
        J_sA[0] = jaccard_score(np.array(Scorearr[:, 1]), np.array(torch.sigmoid(Scorearr[:, 0])).round(),
                                average='macro')
        J_sS[0] = jaccard_score(np.array(labelsA.float().cpu()), np.array(torch.sigmoid(outs)).round(), average='macro')
        J_sS[1] = jaccard_score(np.array(labelsA.float().cpu()), np.array(outs2).round(), average='macro')
        seq_acc[1] = float(torch.sum(outs2.round() == labelsA.float().cpu())) / len(labelsA)
        fpr, tpr, _ = roc_curve(np.array(Scorearr[:, 1]), np.array(torch.sigmoid(Scorearr[:, 0])),
                                pos_label=1)
        aucA[0] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(np.array(labelsA.float().cpu()), np.array(torch.sigmoid(outs)), pos_label=1)
        aucS[0] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(np.array(labelsA.float().cpu()), np.array(outs2), pos_label=1)
        aucS[1] = auc(fpr, tpr)
        print(seq_acc[0])

        reslog(
            [phase, 0, args.ims, args.nch, args.nlayer, args.Dil, args.tt, args.EnbB, args.epochR, args.SL, args.BS,
             args.NN, args.crp, args.lr, args.Sc, args.scSt, args.An, args.JobId,
             args.Ra, args.dd, args.lrW,
             args.Att,
             args.L1, args.L2, args.L3, args.L4, args.D, args.blck, args.opt, args.ims, args.ScR, args.hidS, tr_bb,
             best_acc,
             bep, args.optf, args.NoD, args.ccut, args.prT, epoch_loss[0], 111,
             preA[0], recA[0], fsA[0], J_sA[0], aucA[0], t_a[0], 222, preS[1], recS[1], fsS[1], J_sS[1], aucS[1],
             seq_acc[1], 333, preS[0], recS[0], fsS[0], J_sS[0], aucS[0], seq_acc[0], 555, be_loss_tr, be_loss_v])

        with h5py.File(os.path.join(save_path, 'Ra_{}_day_{}_'.format(args.Ra, 0) + phase + '_features.h5'),
                       'w') as h5f:
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset("ID", data=np.asarray(IDL, dtype=dt))
            h5f.create_dataset("Label", data=np.hstack(labelsA))
            h5f.create_dataset("Features", data=np.stack(feat, axis=0))
            h5f.create_dataset("Out1", data=np.stack(outs.numpy()))
            h5f.create_dataset("Len", data=np.asarray(LenL))
