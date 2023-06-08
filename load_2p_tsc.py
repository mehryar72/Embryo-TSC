# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:27:35 2019

@author: mabbasib
"""

import argparse
import logging

from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
parser.add_argument("--JobId3", type=int, default=95)
parser.add_argument("--JobId5", type=int, default=94)

parser.add_argument("--NN", type=int, default=7)

parser.add_argument("--crp", type=int, default=1)
parser.add_argument("--lr3", type=float, default=0.000001)
parser.add_argument("--lr5", type=float, default=0.000001)

parser.add_argument("--lrW", type=int, default=2)
parser.add_argument("--L1", type=int, default=1)
parser.add_argument("--L2", type=int, default=1)
parser.add_argument("--L3", type=int, default=2)
parser.add_argument("--L4", type=int, default=2)
parser.add_argument("--D", type=int, default=8)
parser.add_argument("--D2", type=int, default=2)
parser.add_argument("--Frz", type=int, default=1)
parser.add_argument("--mode3", type=int, default=400)
parser.add_argument("--mode5", type=int, default=1408)
parser.add_argument("--out0", type=int, default=0)
parser.add_argument("--nlayer", type=int, default=3)
parser.add_argument("--Ra", type=int, default=3)
parser.add_argument("--blck", type=int, default=1)
parser.add_argument("--Att", type=int, default=0)
parser.add_argument("--opt", type=int, default=0)
parser.add_argument("--ims", type=int, default=224)
parser.add_argument("--ScR", type=int, default=0)
parser.add_argument("--hidS3", type=int, default=128)
parser.add_argument("--hidS5", type=int, default=64)
parser.add_argument("--optf", type=int, default=0)
parser.add_argument("--NoD", type=int, default=1)
parser.add_argument("--ccut", type=int, default=1)
parser.add_argument("--Dil", type=int, default=1)
parser.add_argument("--DilForm", type=int, default=0)

parser.add_argument("--epochR", type=int, default=5)
parser.add_argument("--EnbB", type=int, default=0)
parser.add_argument("--prT", type=int, default=1)
parser.add_argument("--nch", type=int, default=8)
parser.add_argument("--adaD3", type=int, default=3)
parser.add_argument("--adaD5", type=int, default=1)
parser.add_argument("--pos_enc3", type=int, default=0)
parser.add_argument("--pos_enc5", type=int, default=2)
parser.add_argument("--lrMo3", type=int, default=2)
parser.add_argument("--lrMo5", type=int, default=2)
parser.add_argument("--mm", type=int, default=0)

parser.add_argument("--day5", type=int, default=1)



args = parser.parse_args()

dd = args.dd



def reslog(line):
    with open(r'./acc_day_2p_tsc_combo_f.csv', 'a', newline='') as f:
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



dictRA5 = {0: 85, 1: 82, 2: 81, 3: 85, 4: 84}
DD5 = 4

dictRA3 = {0: 81, 1: 83, 2: 82, 3: 84, 4: 85}
DD3 = 8

if args.lr3 == 0.0001:
    ppl = 4
elif args.lr3 == 0.00001:
    ppl = 5
else:
    ppl = 6

mod_path3 = (
        'train_day_tsc--day50--JobId{}--Ra{}--mode{}--Dil1--DilForm0--adaD{}--pos_enc{}--lr{' + ':.{}f'.format(
    ppl) + '}--lrMo{}--hidS{}.pth').format(args.JobId3, args.Ra, args.mode3, args.adaD3, args.pos_enc3, args.lr3,
                                           args.lrMo3, args.hidS3)


if args.lr5 == 0.0001:
    ppl = 4
elif args.lr5 == 0.00001:
    ppl = 5
else:
    ppl = 6
mod_path5 = (
        'train_day_tsc--day51--JobId{}--Ra{}--mode{}--Dil1--DilForm0--adaD{}--pos_enc{}--lr{' + ':.{}f'.format(
    ppl) + '}--lrMo{}--hidS{}.pth').format(args.JobId5, args.Ra, args.mode5, args.adaD5, args.pos_enc5, args.lr5,
                                           args.lrMo5, args.hidS5)


save_path = './EmbyroFeatures/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt_path = "Data_c2_S1R_opf2/"
length_paths = "./out.csv"
Cord_path = "./dictCord.pkl"
root_dir = './D_ac/'

day_index_path5 = './ix96.csv'

day_index_path3 = './ix48.csv'

seq_length = args.SL
NW = 0
# batchS = int(104 * 1.0 / args.BS)
batchS = args.BS
ratio = [0.85, 0.15, 0.0]
best_acc = 0.0
Sc = args.Sc
num_epochs = 1




net3 = CNNTSC_TSC(FC=1024, n_cl=1, D=DD3, blck=args.blck, layers=[args.L1, args.L2, args.L3, args.L4],
                  Att=args.Att, normalize_attn=args.ScR, hidden_size=args.hidS3, optf=args.optf, prT=args.prT,
                  mod_path='', Frz=args.Frz, mode=args.mode3, D2=args.D2, out=args.out0,
                  lenght=int(96 / args.Dil), dilF=args.DilForm, dil=args.Dil, device=device, adaD=args.adaD3,
                  pos_enc=args.pos_enc3, LrMo=args.lrMo3)

pretrained_dict = torch.load(mod_path3, map_location=lambda storage, loc: storage)
net3.load_state_dict(pretrained_dict)

net5 = CNNTSC_TSC(FC=1024, n_cl=1, D=DD5, blck=args.blck, layers=[args.L1, args.L2, args.L3, args.L4],
                  Att=args.Att, normalize_attn=args.ScR, hidden_size=args.hidS5, optf=args.optf, prT=args.prT,
                  mod_path='', Frz=args.Frz, mode=args.mode5, D2=args.D2, out=args.out0,
                  lenght=int(96 / args.Dil), dilF=args.DilForm, dil=args.Dil, device=device, adaD=args.adaD5,
                  pos_enc=args.pos_enc5, LrMo=args.lrMo5)

pretrained_dict = torch.load(mod_path5, map_location=lambda storage, loc: storage)
net5.load_state_dict(pretrained_dict)

net3 = net3.to(device)
net5 = net5.to(device)


epochR = args.epochR

# else:
ValiD3 = MyDataset_pd_tsc(
    data_path=os.path.join(save_path, 'Ra_{}_day_{}_val_features.h5'.format(args.Ra, 0)),
    dil=args.Dil if not args.DilForm else 1, day5=0)

LdV3 = DataLoader(
    ValiD3,
    batch_size=batchS,
    num_workers=NW,
    pin_memory=False
)

ValiD5 = MyDataset_pd_tsc(
    data_path=os.path.join(save_path, 'Ra_{}_day_{}_val_features.h5'.format(args.Ra, 1)),
    dil=args.Dil if not args.DilForm else 1, day5=1)

LdV5 = DataLoader(
    ValiD5,
    batch_size=batchS,
    num_workers=NW,
    pin_memory=False
)

#########################################################################

data_loaders = {"val3": LdV3, "val5": LdV5}
data_lengths = {"val3": ValiD3, "val5": ValiD5}
net = {"val3": net3, "val5": net5}
labelsA3 = []
outs0A3 = []
outs1A3 = []
labelsA5 = []
outs0A5 = []
outs1A5 = []
outs2A3 = []
outs2A5 = []
lbAP = {"val3": labelsA3, "val5": labelsA5}
outs0P = {"val3": outs0A3, "val5": outs0A5}
outs1P = {"val3": outs1A3, "val5": outs1A5}
outs2P = {"val3": outs2A3, "val5": outs2A5}

preS = [None] * 7
recS = [None] * 7
fsS = [None] * 7
J_sS = [None] * 7

aucS = [None] * 7
seq_acc = [None] * 7

for epoch in range(num_epochs):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['val3', 'val5']:

        net[phase].train(False)  # Set model to evaluate mode

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

            with torch.set_grad_enabled(False):
                outputs, out0 = net[phase](images)
                lbAP[phase].extend(labels.cpu().detach().clone())
                outs0P[phase].extend(outputs.cpu().detach().clone())
                outs1P[phase].extend(torch.sigmoid(outputs.cpu().detach().clone()).numpy())
                outs2P[phase].extend(torch.sigmoid(out1.unsqueeze(1)).numpy())

            pred = outputs.data.clone()

            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            prediction = pred
            pred2 = out1.data.clone()
            pred2[pred2 >= 0] = 1
            pred2[pred2 < 0] = 0
            train_acc += torch.sum(prediction.squeeze(1) == labels.float())
            train_acc2 += torch.sum(pred2 == labels.float().cpu().detach().clone())

a = ((np.array(outs2P["val5"]) + np.array(outs2P["val3"])) / 2).round()
b = ((np.array(outs2P["val5"]).round() + np.array(outs2P["val3"])) / 2).round()
c = ((np.array(outs2P["val5"]) + np.array(outs2P["val3"]).round()) / 2).round()
d = ((np.array(outs1P["val5"]).round() + np.array(outs1P["val3"]).round() + np.array(
    outs2P["val3"]).round()) / 3).round()
f = ((np.array(outs1P["val5"]).round() + np.array(outs1P["val3"]).round() + np.array(
    outs2P["val5"]).round()) / 3).round()

preS[0], recS[0], fsS[0], _ = precision_recall_fscore_support(np.array(lbAP["val3"]),
                                                              np.array(outs1P["val3"]).round(),
                                                              average='macro')
preS[1], recS[1], fsS[1], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              np.array(outs1P["val5"]).round(),
                                                              average='macro')

preS[2], recS[2], fsS[2], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              d,
                                                              average='macro')
out0 = torch.sigmoid((torch.stack(outs0P["val3"]) + torch.stack(outs0P["val5"])) / 2).numpy()

preS[3], recS[3], fsS[3], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              c,
                                                              average='macro')
preS[4], recS[4], fsS[4], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              b,
                                                              average='macro')
out0 = torch.sigmoid((torch.stack(outs0P["val3"]) + torch.stack(outs0P["val5"])) / 2).numpy()

preS[5], recS[5], fsS[5], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              a,
                                                              average='macro')

preS[6], recS[6], fsS[6], _ = precision_recall_fscore_support(np.array(lbAP["val5"]),
                                                              f,
                                                              average='macro')

J_sS[0] = jaccard_score(np.array(lbAP["val3"]), np.array(outs1P["val3"]).round(),
                        average='macro')
J_sS[1] = jaccard_score(np.array(lbAP["val5"]), np.array(outs1P["val5"]).round(),
                        average='macro')

J_sS[2] = jaccard_score(np.array(lbAP["val3"]), d,
                        average='macro')
J_sS[3] = jaccard_score(np.array(lbAP["val5"]), c,
                        average='macro')

J_sS[4] = jaccard_score(np.array(lbAP["val3"]), b,
                        average='macro')
J_sS[5] = jaccard_score(np.array(lbAP["val5"]), a,
                        average='macro')

J_sS[6] = jaccard_score(np.array(lbAP["val5"]), f,
                        average='macro')


seq_acc[0] = accuracy_score(np.array(lbAP["val3"]), np.array(outs1P["val3"]).round())
seq_acc[1] = accuracy_score(np.array(lbAP["val5"]), np.array(outs1P["val5"]).round())
seq_acc[2] = accuracy_score(np.array(lbAP["val5"]), d)
seq_acc[3] = accuracy_score(np.array(lbAP["val5"]), c)
seq_acc[4] = accuracy_score(np.array(lbAP["val5"]), b)
seq_acc[5] = accuracy_score(np.array(lbAP["val5"]), a)
seq_acc[6] = accuracy_score(np.array(lbAP["val5"]), f)


reslog(
    [args.mode3, args.mode5, args.adaD3, args.adaD5, args.pos_enc3, args.pos_enc5, args.hidS3, args.hidS5,
     args.lrMo3, args.lrMo5, args.lr3, args.lr5, args.JobId3, args.JobId5,
     args.Ra, 333,
     preS[0], recS[0], fsS[0], J_sS[0], seq_acc[0], 555, preS[1], recS[1], fsS[1], J_sS[1], seq_acc[1], 'ddd', preS[2],
     recS[2], fsS[2], J_sS[2], seq_acc[2], 'ccc', preS[3], recS[3], fsS[3], J_sS[3], seq_acc[3], 'bbb', preS[4],
     recS[4], fsS[4], J_sS[4], seq_acc[4], 'aaa', preS[5], recS[5], fsS[5], J_sS[5], seq_acc[5], 'fff', preS[6],
     recS[6], fsS[6], J_sS[6], seq_acc[6]])
