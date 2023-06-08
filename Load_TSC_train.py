# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:27:35 2019

@author: mabbasib
"""
# import os
# import glob

import argparse
import logging
import sys

import torch.optim.lr_scheduler as lrs
from sklearn.metrics import jaccard_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam
from torch.utils.data import DataLoader

from DataLoad import *
from modelsN import *

parser = argparse.ArgumentParser()
parser.add_argument("--SL", type=int, default=1)
parser.add_argument("--tt", type=int, default=62)

parser.add_argument("--BS", type=int, default=16)
parser.add_argument("--Sc", type=int, default=1)
parser.add_argument("--scSt", type=int, default=0)
parser.add_argument("--An", type=int, default=1)
parser.add_argument("--dd", type=int, default=1)
parser.add_argument("--JobId", type=int, default=11)
parser.add_argument("--JobId0", type=int, default=11)

parser.add_argument("--NN", type=int, default=7)

parser.add_argument("--crp", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--lr0", type=float, default=0.001)

parser.add_argument("--lrW", type=int, default=2)
parser.add_argument("--L1", type=int, default=1)
parser.add_argument("--L2", type=int, default=1)
parser.add_argument("--L3", type=int, default=2)
parser.add_argument("--L4", type=int, default=2)
parser.add_argument("--D", type=int, default=8)
parser.add_argument("--D2", type=int, default=2)
parser.add_argument("--Frz", type=int, default=1)
parser.add_argument("--mode", type=int, default=400)
parser.add_argument("--out0", type=int, default=0)
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
parser.add_argument("--DilForm", type=int, default=0)

parser.add_argument("--epochR", type=int, default=5)
parser.add_argument("--EnbB", type=int, default=0)
parser.add_argument("--prT", type=int, default=1)
parser.add_argument("--nch", type=int, default=8)
parser.add_argument("--adaD", type=int, default=1)
parser.add_argument("--pos_enc", type=int, default=2)
parser.add_argument("--lrMo", type=int, default=1)
parser.add_argument("--mm", type=int, default=0)

parser.add_argument("--day5", type=int, default=1)

args = parser.parse_args()

dd = args.dd


def save_models(epoch):
    torch.save(net1.state_dict(), netS)


def reslog(line):
    with open(r'./acc.csv', 'a', newline='') as f:
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

if args.day5:

    if args.mm == 1:
        mm = 8112212
        dictRA = {0: 81, 1: 85, 2: 83, 3: 84, 4: 85}
        DD = 8
    elif args.mm == 2:
        mm = 4112212
        dictRA = {0: 83, 1: 83, 2: 85, 3: 84, 4: 83}
        DD = 4
    elif args.mm == 3:
        mm = 4111105
        dictRA = {0: 83, 1: 83, 2: 81, 3: 82, 4: 82}
        DD = 4
    else:

        dictRA = {0: 85, 1: 82, 2: 81, 3: 85, 4: 84}
        DD = 4
else:
    if args.mm == 1:
        mm = 8111105
        dictRA = {0: 82, 1: 85, 2: 83, 3: 84, 4: 82}
        DD = 8
    elif args.mm == 2:
        mm = 16112205
        dictRA = {0: 82, 1: 82, 2: 83, 3: 81, 4: 81}
        DD = 16
    else:
        dictRA = {0: 81, 1: 83, 2: 82, 3: 84, 4: 85}
        DD = 8

name = './train_day_tsc' + (''.join(sys.argv[1:]))
if args.day5:
    old_name = './rundataDay5_r/train_day--Ra{}--NN9--JobId{}--lr{}--L11--L21--L32--L42--blck1--D4--prT0--nch64--ims224--nlayer4--EnbB0--epochR5.pth'.format(
        args.Ra, dictRA[args.Ra], args.lr0)

else:
    old_name = './rundataDay3_r/train_day--Ra{}--NN9--JobId{}--lr{}--L11--L21--L32--L42--blck1--D8--prT0--nch64--ims224--nlayer4.pth'.format(
        args.Ra, dictRA[args.Ra], args.lr0)

logF = name + '.csv'
netS = name + '.pth'
save_path = './EmbyroFeatures/'

if (os.path.isfile(logF)):
    os.remove(logF)
logging.basicConfig(filename=logF,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('myloger')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt_path = "./Data_c2_S1R_opf2/"
length_paths = "./out.csv"
Cord_path = "./dictCord.pkl"
root_dir = './D_ac/'

if args.day5:
    day_index_path = './ix96.csv'
else:
    day_index_path = './ix48.csv'

seq_length = args.SL
NW = 0

batchS = args.BS
ratio = [0.85, 0.15, 0.0]
best_acc = 0.0
Sc = args.Sc
num_epochs = 1000 if args.dd else 1000

if args.mode == 0:
    hids = int(args.hidS / args.D2)
else:
    hids = args.hidS

if args.NN == 97:
    net1 = CNNTSC(FC=1024, n_cl=1, D=args.D, blck=args.blck, layers=[args.L1, args.L2, args.L3, args.L4],
                  Att=args.Att, normalize_attn=args.ScR, hidden_size=hids, optf=args.optf, prT=args.prT,
                  mod_path=old_name, Frz=args.Frz, mode=args.mode, D2=args.D2, out=args.out0,
                  lenght=int(96 / args.Dil), dilF=args.DilForm, dil=args.Dil, device=device, adaD=args.adaD,
                  pos_enc=args.pos_enc)
elif args.NN == 7:
    net1 = CNNTSC_TSC(FC=1024, n_cl=1, D=DD, blck=args.blck, layers=[args.L1, args.L2, args.L3, args.L4],
                      Att=args.Att, normalize_attn=args.ScR, hidden_size=hids, optf=args.optf, prT=args.prT,
                      mod_path=old_name, Frz=args.Frz, mode=args.mode, D2=args.D2, out=args.out0,
                      lenght=int(96 / args.Dil), dilF=args.DilForm, dil=args.Dil, device=device, adaD=args.adaD,
                      pos_enc=args.pos_enc, LrMo=args.lrMo)

print(count_parameters(net1))
logger.info(count_parameters(net1))
print(list(net1.children()))
net1 = net1.to(device)
optimizer = Adam(filter(lambda p: p.requires_grad, net1.parameters()), lr=args.lr, weight_decay=10 ** (-args.lrW))

print(10 ** -args.lrW)

loss_fn = torch.nn.BCEWithLogitsLoss()
if args.scSt:
    scheduler = lrs.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
else:
    scheduler = lrs.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, cooldown=1, min_lr=0.000001)

epochR = args.epochR
if not args.mm == 0:
    TrainD = MyDataset_pd_tsc(
        data_path=os.path.join(save_path, 'Ra_{}_day_{}_Mode_{}train_features.h5'.format(args.Ra, args.day5, mm)),
        dil=args.Dil if not args.DilForm else 1, day5=args.day5)
else:
    TrainD = MyDataset_pd_tsc(
        data_path=os.path.join(save_path, 'Ra_{}_day_{}_train_features.h5'.format(args.Ra, args.day5)),
        dil=args.Dil if not args.DilForm else 1, day5=args.day5)

LdT = DataLoader(
    TrainD,
    batch_size=batchS,
    num_workers=NW,
    pin_memory=False
)
if not args.mm == 0:
    ValiD = MyDataset_pd_tsc(
        data_path=os.path.join(save_path, 'Ra_{}_day_{}_Mode_{}val_features.h5'.format(args.Ra, args.day5, mm)),
        dil=args.Dil if not args.DilForm else 1, day5=args.day5)
else:
    ValiD = MyDataset_pd_tsc(
        data_path=os.path.join(save_path, 'Ra_{}_day_{}_val_features.h5'.format(args.Ra, args.day5)),
        dil=args.Dil if not args.DilForm else 1, day5=args.day5)

LdV = DataLoader(
    ValiD,
    batch_size=batchS,
    num_workers=NW,
    pin_memory=False
)

#########################################################################

data_loaders = {"train": LdT, "val": LdV}
data_lengths = {"train": TrainD, "val": ValiD}
tloss = 0
vloss = 0
tac = 0
vac = 0
lgloss = {"train": tloss, "val": vloss}
lgacc = {"train": tac, "val": vac}
best_acc = 0
cc = 0
inv = 0
bep = 0
epochi = 1
be_loss_v = 0
be_loss_tr = 0
preS = [None] * 2
recS = [None] * 2
fsS = [None] * 2
J_sS = [None] * 2
# aucA = [None] * 1
aucS = [None] * 2
seq_acc = [None] * 2

for epoch in range(num_epochs):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            net1.train(True)  # Set model to training mode
        else:
            labelsA = []
            outs0A = []
            outs1A = []
            net1.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        train_acc = 0
        train_acc2 = 0
        t_a_e = 0
        for data in data_loaders[phase]:

            images = data[0]
            labels = data[1]
            lens = data[3]
            optf = data[4]
            out1 = data[-2]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                if phase == 'train':
                    outputs, out0 = net1(images)
                else:

                    outputs, out0 = net1(images)
                    labelsA.extend(labels.cpu().detach().clone())
                    outs0A.extend(torch.sigmoid(out1.cpu().detach().clone()).numpy())
                    outs1A.extend(torch.sigmoid(outputs.cpu().detach().clone()).numpy())

                loss = loss_fn(outputs.squeeze(1), labels.float())

                if phase == 'train':
                    loss.backward()

                    optimizer.step()

            running_loss += loss.data.item() * float(len(data[1]))

            pred = outputs.data.clone()

            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            prediction = pred
            pred2 = out1.data.clone()
            pred2[pred2 >= 0] = 1
            pred2[pred2 < 0] = 0
            train_acc += torch.sum(prediction.squeeze(1) == labels.float())
            train_acc2 += torch.sum(pred2 == labels.float().cpu().detach().clone())

            print(float(torch.sum(prediction.squeeze(1) == labels.float())) / float(len(data[1])))
            print(float(torch.sum(pred2 == labels.float().cpu().detach().clone())) / float(len(data[1])))
            print("--" * 20)
        epoch_loss = float(running_loss) / float(len(data_lengths[phase]))
        if ((Sc == 1) & (phase == 'val')) & (epoch >= 0):
            if args.scSt:
                scheduler.step(epoch)
            else:
                scheduler.step(epoch_loss)

        t_a = float(train_acc) / float(len(data_lengths[phase]))
        t_a2 = float(train_acc2) / float(len(data_lengths[phase]))
        print('{} Loss: {:} Acc:{} Acc2:{}'.format(phase, epoch_loss, t_a, t_a2))

        lgloss[phase] = epoch_loss
        lgacc[phase] = t_a
        if phase == 'val':
            print('best_ac:{}'.format(best_acc))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

            if (((t_a - best_acc) >= 0.001) or (t_a >= best_acc) and (be_loss_v >= lgloss["val"])) and epoch >= 0:
                save_models(epoch)
                bep = epoch
                print((t_a - best_acc))

                logger.info(
                    ',{},{},{},{},{},{},{},{}'.format(lgloss["train"], lgloss["val"], lgacc["train"], lgacc["val"],
                                                      best_acc, t_a2, (t_a - best_acc), param_group['lr']))
                best_acc = t_a
                tr_bb = tr_b
                be_loss_v = lgloss["val"]
                be_loss_tr = lgloss["train"]
                cc = 0

                preS[0], recS[0], fsS[0], _ = precision_recall_fscore_support(np.array(labelsA),
                                                                              np.array(outs0A).round(),
                                                                              average='macro')
                preS[1], recS[1], fsS[1], _ = precision_recall_fscore_support(np.array(labelsA),
                                                                              np.array(outs1A).round(),
                                                                              average='macro')

                J_sS[0] = jaccard_score(np.array(labelsA), np.array(outs0A).round(),
                                        average='macro')
                J_sS[1] = jaccard_score(np.array(labelsA), np.array(outs1A).round(),
                                        average='macro')

                seq_acc[1] = t_a
                seq_acc[0] = t_a2

                fpr, tpr, _ = roc_curve(np.array(labelsA), np.array(outs0A).round(), pos_label=1)
                aucS[0] = auc(fpr, tpr)
                fpr, tpr, _ = roc_curve(np.array(labelsA), np.array(outs1A).round(), pos_label=1)
                aucS[1] = auc(fpr, tpr)

            else:
                logger.info(
                    ',{},{},{},{},{},{},{}'.format(lgloss["train"], lgloss["val"], lgacc["train"], lgacc["val"],
                                                   best_acc, t_a2,
                                                   param_group['lr']))
                if epoch > 10:
                    cc = cc + 1
                if cc > num_epochs:
                    break
        else:
            tr_b = t_a

    else:
        # Continue if the inner loop wasn't broken.
        continue
        # Inner loop was broken, break the outer.

    break

logging.shutdown()
#########


if not args.mm == 0:
    reslog(
        [args.day5, args.mm, args.adaD, args.pos_enc, args.hidS, args.lrMo, args.D2, args.Frz, args.mode, args.out0,
         args.nch, args.nlayer, args.Dil, args.DilForm, args.tt, args.EnbB,
         args.epochR, args.SL, args.BS, args.NN, args.crp, args.lr, args.lr0, args.Sc, args.scSt, args.An,
         args.JobId, args.JobId0,
         args.Ra, args.dd, args.lrW,
         args.Att,
         args.L1, args.L2, args.L3, args.L4, args.D, args.blck, args.opt, args.ims, args.ScR, args.hidS, tr_bb,
         best_acc,
         bep, 555, be_loss_tr, be_loss_v, 555, args.optf, args.NoD, args.ccut, args.prT, epoch_loss, 222,
         preS[0], recS[0], fsS[0], J_sS[0], aucS[0],
         seq_acc[0], 333, preS[1], recS[1], fsS[1], J_sS[1], aucS[1], seq_acc[1], 0, 0, count_parameters(net1)])
else:
    reslog(
        [args.day5, args.adaD, args.pos_enc, args.hidS, args.lrMo, args.D2, args.Frz, args.mode, args.out0, args.nch,
         args.nlayer, args.Dil, args.DilForm, args.tt, args.EnbB,
         args.epochR, args.SL, args.BS, args.NN, args.crp, args.lr, args.lr0, args.Sc, args.scSt, args.An,
         args.JobId, args.JobId0,
         args.Ra, args.dd, args.lrW,
         args.Att,
         args.L1, args.L2, args.L3, args.L4, args.D, args.blck, args.opt, args.ims, args.ScR, args.hidS, tr_bb,
         best_acc,
         bep, 555, be_loss_tr, be_loss_v, 555, args.optf, args.NoD, args.ccut, args.prT, epoch_loss, 222,
         preS[0], recS[0], fsS[0], J_sS[0], aucS[0],
         seq_acc[0], 333, preS[1], recS[1], fsS[1], J_sS[1], aucS[1], seq_acc[1], 0, 0, count_parameters(net1)])
