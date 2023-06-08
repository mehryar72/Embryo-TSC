# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:27:35 2019

@author: mabbasib
"""
# import os
# import glob

import argparse
# from torch.optim import RMSprop
import sys

import torchvision.transforms as transforms
# from AdamW import AdamW
from torch.utils.data import DataLoader

from DataLoad import *
from modelsN import *

# from triplet_loss import *

# from tta import *


parser = argparse.ArgumentParser()
parser.add_argument("--SL", type=int, default=1)
parser.add_argument("--tt", type=int, default=62)
# parser.add_argument("--Fix", type=int, default=1)
parser.add_argument("--BS", type=int, default=16)
parser.add_argument("--Sc", type=int, default=1)
parser.add_argument("--scSt", type=int, default=0)
parser.add_argument("--An", type=int, default=1)
parser.add_argument("--dd", type=int, default=1)
parser.add_argument("--JobId", type=int, default=11)
parser.add_argument("--JobId0", type=int, default=11)
# parser.add_argument("--JobIdA", type=int, default=51)
parser.add_argument("--NN", type=int, default=90)
# parser.add_argument("--NNA", type=int, default=1145)
parser.add_argument("--crp", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--lr0", type=float, default=0.0001)
# parser.add_argument("--lrA", type=float, default=0.001)
parser.add_argument("--lrW", type=int, default=2)
parser.add_argument("--L1", type=int, default=1)
parser.add_argument("--L2", type=int, default=1)
parser.add_argument("--L3", type=int, default=2)
parser.add_argument("--L4", type=int, default=2)
parser.add_argument("--D", type=int, default=8)
parser.add_argument("--D2", type=int, default=1)
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
# parser.add_argument("--lenR", type=int, default=1)
parser.add_argument("--epochR", type=int, default=5)
parser.add_argument("--EnbB", type=int, default=0)
parser.add_argument("--prT", type=int, default=1)
parser.add_argument("--nch", type=int, default=8)
parser.add_argument("--adaD", type=int, default=1)
parser.add_argument("--pos_enc", type=int, default=2)
parser.add_argument("--day5", type=int, default=1)
# parser.add_argument("--prTA", type=int, default=1)

args = parser.parse_args()

print(args.Ra)
print(args.day5)

dd = args.dd


def reslog(line):
    with open(r'/home/mabbasib/EmbDayTSC2/accs/acc_eval.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()


if args.day5:
    dictRA = {0: 83, 1: 85, 2: 83, 3: 81, 4: 81}
else:
    dictRA = {0: 81, 1: 81, 2: 83, 3: 83, 4: 82}

print(dictRA[args.Ra])
print(args.lr0)
# dict={0:83,1:82,2:81,3:82,4:82}
name = '/scratch/mabbasib/rundatatday3_newTSC/train_day_tsc' + (''.join(sys.argv[1:]))
if args.day5:
    old_name = '/scratch/mabbasib/rundataDay5_r/train_day--Ra{}--NN9--JobId{}--lr{}--L11--L21--L32--L42--blck1--D4--prT0--nch64--ims224--nlayer4--EnbB0--epochR5.pth'.format(
        args.Ra, dictRA[args.Ra], args.lr0)

else:
    old_name = '/scratch/mabbasib/rundataDay3_r/train_day--Ra{}--NN9--JobId{}--lr{}--L11--L21--L32--L42--blck1--D8--prT0--nch64--ims224--nlayer4.pth'.format(
        args.Ra, dictRA[args.Ra], args.lr0)
print(old_name)
# old_name ='logs\\train_day3.pth'
logF = name + '.csv'
netS = name + '.pth'
save_path = '/scratch/mabbasib/EmbyroFeatures/'

# name = 'logs\\train_day_tsc' + (''.join(sys.argv[1:]))
# old_name = 'logs\\train_day--Ra{}--NN9--JobId{}--lr{}--L11--L21--L32--L42--blck1--D8--prT0--nch64--ims224--nlayer4.pth'.format(args.Ra,args.JobId0,args.Lr0)
# old_name = 'logs\\train_day3.pth'
# save_path ='logs\\'
# logF = name + '.csv'
# netS = name + '.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt_path = "/home/mabbasib/Data_c2_S1R_opf2/"
length_paths = "/home/mabbasib/out.csv"
Cord_path = "/home/mabbasib/dictCord.pkl"
root_dir = '/home/mabbasib/D_ac/'
# root_dir = './D_ac/'


if args.day5:
    day_index_path = '/home/mabbasib/embroDay/ix96.csv'
else:
    day_index_path = '/home/mabbasib/embroDay/ix48.csv'

seq_length = args.SL
NW = 0

batchS = args.BS
ratio = [0.85, 0.15, 0.0]
best_acc = 0.0
Sc = args.Sc
num_epochs = 100 if args.dd else 100

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
elif args.NN == 90:
    net1 = CNNTSC_CNN(FC=1024, n_cl=1, D=4 if args.day5 else 8, blck=args.blck,
                      layers=[args.L1, args.L2, args.L3, args.L4],
                      Att=args.Att, normalize_attn=args.ScR, hidden_size=hids, optf=args.optf, prT=args.prT,
                      mod_path=old_name, Frz=args.Frz, mode=args.mode, D2=args.D2, out=args.out0,
                      lenght=int(96 / args.Dil), dilF=args.DilForm, dil=args.Dil, device=device, adaD=args.adaD,
                      pos_enc=args.pos_enc)
# elif args.NN    Att=args.Att, normalize_attn=args.ScR, hidden_size=hids, optf=args.optf, prT=args.prT,mod_path=old_name,Frz=args.Frz,mode=args.mode,D2=args.D2,out=args.out0,lenght=int(96/args.Dil),dilF=args.DilForm)
print(count_parameters(net1))
print(list(net1.children()))
net1 = net1.to(device)

c_p1, c_p2, e_i1, e_i2 = datPP3(root_dir, args.Ra)
torch.save({'c_p1': c_p1, 'c_p2': c_p2, 'e_i1': e_i1, 'e_i2': e_i2},
           'c_p11_Ra_{}_day_{}.pth'.format(args.Ra, args.day5))

p = 1
transform = transforms.Compose([

    # transforms.RandomRotation(args.An * 45, resample=Image.BICUBIC),
    transforms.RandomAffine(args.An * 45, translate=(0.05, 0.05), scale=None, shear=2, resample=Image.NEAREST,
                            fillcolor=102 if args.crp else 0),
    transforms.ColorJitter(brightness=0.5, contrast=0.05),
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
    # Crp(c=args.crp),
    # transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.05), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    # RepC(p=1),
    transforms.Normalize([0.4], [0.2]),
    RepC(p=p)
])

transformV = transforms.Compose([
    # Crp(c=args.crp),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # RepC(p=1),
    transforms.Normalize([0.4],
                         [0.2]),
    RepC(p=p),

])

epochR = args.epochR

samTr = MySamplerDilate(e_i1, 1, 1)
samVa = MySamplerDilate(e_i2, 1, 1)

TrainD = MyDatasetDivideNrep_Day_tsc(
    image_paths=c_p1,
    seq_length=seq_length,
    transform=transformV,
    length=len(samTr),
    end_idx=e_i1,
    C_path=Cord_path,
    L_path=length_paths,
    O_path=opt_path,
    st_path=day_index_path,
    crpc=args.crp, opt=args.optf, NoD=args.NoD, cut=args.ccut, dil=args.Dil if not args.DilForm else 1, day5=args.day5)
LdT = DataLoader(
    TrainD,
    batch_size=batchS,
    sampler=samTr,
    num_workers=NW,
    pin_memory=False
)

ValiD = MyDatasetDivideNrep_Day_tsc(
    image_paths=c_p2,
    seq_length=seq_length,
    transform=transformV,
    # length=len(e_i2)-1,
    length=len(samVa),
    end_idx=e_i2,
    C_path=Cord_path,
    L_path=length_paths,
    st_path=day_index_path,
    O_path=opt_path, crpc=args.crp, opt=args.optf, NoD=args.NoD, cut=args.ccut, dil=args.Dil if not args.DilForm else 1,
    day5=args.day5)

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
cc = 0
inv = 0
bep = 0
epochi = 1
be_loss_v = 0
be_loss_tr = 0

for epoch in range(1):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    #     epochi += 1
    for phase in ['train', 'val']:
        Id_list = []
        Label_list = []
        Features_List = []
        Out1_List = []
        Len_List = []
        net1.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        train_acc = 0
        train_acc2 = 0
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
            images = images.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):

                if phase == 'train':
                    outputs, out0, out1 = net1(images, lens, optf)
                else:

                    outputs, out0, out1 = net1(images, lens, optf)

            pred2 = out1.data.clone()
            pred2[pred2 >= 0] = 1
            pred2[pred2 < 0] = 0
            train_acc2 += torch.sum(pred2 == labels.float())

            print(float(torch.sum(pred2 == labels.float())) / float(len(data[1])))
            print("--" * 20)
            # for cc,sample in enumerate(outputs):
            Id_list.extend(data[5])
            Label_list.extend(data[1].numpy())
            Features_List.extend(outputs.cpu().numpy())
            Len_List.extend(data[3].numpy())
            Out1_List.extend(out0.cpu().numpy())
        with h5py.File(os.path.join(save_path, 'Ra_{}_day_{}_'.format(args.Ra, args.day5) + phase + '_features.h5'),
                       'w') as h5f:
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset("ID", data=np.asarray(Id_list, dtype=dt))
            h5f.create_dataset("Label", data=np.hstack(Label_list))
            h5f.create_dataset("Features", data=np.stack(Features_List, axis=0))
            h5f.create_dataset("Out1", data=np.stack(Out1_List))
            h5f.create_dataset("Len", data=np.asarray(Len_List))

        t_a = float(train_acc) / float(len(data_lengths[phase]))
        t_a2 = float(train_acc2) / float(len(data_lengths[phase]))
        print('{} Acc:{} Acc2:{}'.format(phase, t_a, t_a2))
        reslog([phase, args.Ra, dictRA[args.Ra], args.lr0, args.day5, t_a, t_a2])
