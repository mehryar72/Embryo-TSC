import csv
import glob
import os
import pickle
import random
from itertools import islice
from pathlib import Path
from random import shuffle

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


class MyDatasetDivideNrep_Day_order(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=2,
                 cut=1, end=96):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        self.st_path = st_path
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            # start = self.end_idx[i]
            # binn = i + 1
            # end = self.end_idx[binn]
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = start + self.end
            chunks = np.arange(start, end)
            # chunks=np.array([start,start,start])

            # lens=[]
            # np.random.shuffle(chunks)
            # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))

            self.it.append(chunks)
        self.test = []

    # def updateD(self):
    #     self.it = []
    #     for i in range(len(self.end_idx) - 1):
    #
    #         init = self.end_idx[i]
    #         start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
    #         end = start + self.end
    #         chunks = np.arange(start, end)
    #         np.random.shuffle(chunks)
    #         self.it.append(chunks)
    #     self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        indices = torch.tensor([self.it[index][0]])
        self.it[index] = np.roll(self.it[index], -1, axis=0)

        # indices = self.it[index]
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h_n = image.shape[0] - y - 1
                else:
                    h_n = h
                if x + w > image.shape[1]:
                    w_n = image.shape[1] - x - 1
                else:
                    w_n = w
                ROI = image[y:y + h_n, x:x + w_n]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h_n > mask.shape[0]:
                    h_n = mask.shape[0] - y1 - 1
                else:
                    h_n = h_n
                if x1 + w_n > mask.shape[1]:
                    w_n = mask.shape[1] - x1 - 1
                else:
                    w_n = w_n
                mask[y1:y1 + h_n, x1:x1 + w_n] = ROI[:h_n, :w_n]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o, id

    def __len__(self):
        return self.length


class MyDatasetDivideNrep_Day5_order(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=2,
                 cut=1, end=96):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        self.st_path = st_path
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            # start = self.end_idx[i]
            # binn = i + 1
            # end = self.end_idx[binn]
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = self.end_idx[i + 1]
            chunks = np.arange(start, end)
            # chunks=np.array([start,start,start])
            # lens=[]
            # np.random.shuffle(chunks)
            # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))

            self.it.append(chunks)
        self.test = []

    # def updateD(self):
    #     self.it = []
    #     for i in range(len(self.end_idx) - 1):
    #
    #         init = self.end_idx[i]
    #         start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
    #         end = self.end_idx[i+1]
    #         chunks = np.arange(start, end)
    #         np.random.shuffle(chunks)
    #         self.it.append(chunks)
    #     self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]

        indices = torch.tensor([self.it[index][0]])
        # indices = self.it[index]
        self.it[index] = np.roll(self.it[index], -1, axis=0)
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h_n = image.shape[0] - y - 1
                else:
                    h_n = h
                if x + w > image.shape[1]:
                    w_n = image.shape[1] - x - 1
                else:
                    w_n = w
                ROI = image[y:y + h_n, x:x + w_n]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h_n > mask.shape[0]:
                    h_n = mask.shape[0] - y1 - 1
                else:
                    h_n = h_n
                if x1 + w_n > mask.shape[1]:
                    w_n = mask.shape[1] - x1 - 1
                else:
                    w_n = w_n
                mask[y1:y1 + h_n, x1:x1 + w_n] = ROI[:h_n, :w_n]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o, id

    def __len__(self):
        return self.length


class RepC(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=3):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: repeated.
        """
        return img.repeat(self.p, 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Crp(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, c=1):
        size = (400, 397)
        self.mask = Image.new('L', (431, 431), 0)
        self.cm = Image.new('L', (431, 431), 102)
        draw = ImageDraw.Draw(self.mask)
        draw.ellipse((31, 24) + size, fill=255)
        self.c = c

    def __call__(self, img):
        if img.size[1] != 431:
            a = Image.new(img.mode, (431, 431))
            a.paste(img, (0, 431 - img.size[1]))
            img = a

        # im = ImageChops.multiply(img, self.mask)
        im = Image.composite(img, self.cm, self.mask)
        im = im.crop((8, 8, 423, 423))
        # return ImageOps.equalize(im)
        # im.show()
        if self.c == 1:
            return im
        else:
            return img.crop((8, 8, 423, 423))
    # def __init__(self):
    #     size=(420,410)
    #     self.mask = Image.new('L', (431, 431), 0)
    #     draw = ImageDraw.Draw(self.mask)
    #     draw.ellipse((31, 21) + size, fill=255)
    #
    # def __call__(self, img):
    #     im = ImageChops.multiply(img, self.mask)
    #     im =im.crop((8,8,423,423))
    #     # return ImageOps.equalize(im)
    #     return im
    # def __repr__(self):
    #     return self.__class__.__name__ + '(p={})'.format(self.mask)


def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#         'Random Erasing Data Augmentation' by Zhong et al.
#         See https://arxiv.org/pdf/1708.04896.pdf
#     Args:
#          p: probability that the random erasing operation will be performed.
#          scale: range of proportion of erased area against input image.
#          ratio: range of aspect ratio of erased area.
#          value: erasing value. Default is 0. If a single int, it is used to
#             erase all pixels. If a tuple of length 3, it is used to erase
#             R, G, B channels respectively.
#             If a str of 'random', erasing each pixel with random values.
#          inplace: boolean to make this transform inplace. Default set to False.
#
#     Returns:
#         Erased Image.
#     # Examples:
#         >>> transform = transforms.Compose([
#         >>> transforms.RandomHorizontalFlip(),
#         >>> transforms.ToTensor(),
#         >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         >>> transforms.RandomErasing(),
#         >>> ])
#     """
#
#     def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
#         assert isinstance(value, (numbers.Number, str, tuple, list))
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("range should be of kind (min, max)")
#         if scale[0] < 0 or scale[1] > 1:
#             raise ValueError("range of scale should be between 0 and 1")
#         if p < 0 or p > 1:
#             raise ValueError("range of random erasing probability should be between 0 and 1")
#
#         self.p = p
#         self.scale = scale
#         self.ratio = ratio
#         self.value = value
#         self.inplace = inplace
#
#     @staticmethod
#     def get_params(img, scale, ratio, value=0):
#         """Get parameters for ``erase`` for a random erasing.
#
#         Args:
#             img (Tensor): Tensor image of size (C, H, W) to be erased.
#             scale: range of proportion of erased area against input image.
#             ratio: range of aspect ratio of erased area.
#
#         Returns:
#             tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
#         """
#         img_c, img_h, img_w = img.shape
#         area = img_h * img_w
#
#         for attempt in range(10):
#             erase_area = random.uniform(scale[0], scale[1]) * area
#             aspect_ratio = random.uniform(ratio[0], ratio[1])
#
#             h = int(round(math.sqrt(erase_area * aspect_ratio)))
#             w = int(round(math.sqrt(erase_area / aspect_ratio)))
#
#             if h < img_h and w < img_w:
#                 i = random.randint(0, img_h - h)
#                 j = random.randint(0, img_w - w)
#                 if isinstance(value, numbers.Number):
#                     v = value
#                 elif isinstance(value, torch._six.string_classes):
#                     v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
#                 elif isinstance(value, (list, tuple)):
#                     v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
#                 return i, j, h, w, v
#
#         # Return original image
#         return 0, 0, img_h, img_w, img
#
#     def __call__(self, img):
#         """
#         Args:
#             img (Tensor): Tensor image of size (C, H, W) to be erased.
#
#         Returns:
#             img (Tensor): Erased Tensor image.
#         """
#         if random.uniform(0, 1) < self.p:
#             x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
#             return F.erase(img, x, y, h, w, v, self.inplace)
#         return img


############################
class MySamplerALLALL(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []
        for i in range(len(end_idx[0]) - 1):
            L = []

            for e_i in end_idx:
                L.append(e_i[i + 1] - e_i[i])
            repeats = int(np.prod(np.array(L)))
            indices += repeats * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1]

            if (((end_idx[i + 1] - end_idx[i]) % seq_length)):
                if ((end_idx[i + 1] - end_idx[i]) < seq_length):
                    indices.append(torch.arange(start, end, seq_length, dtype=torch.int64))
                else:
                    indices.append(torch.cat((torch.arange(start, end - seq_length, seq_length, dtype=torch.int64),
                                              torch.tensor([end - seq_length]).view(1)), 0))

            else:
                indices.append(torch.arange(start, end, seq_length, dtype=torch.int64))

        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySamplerEND(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1]

            if (((end_idx[i + 1] - end_idx[i]) % seq_length)):
                if ((end_idx[i + 1] - end_idx[i]) < seq_length):
                    indices.append(torch.arange(start, end, seq_length, dtype=torch.int64))
                else:
                    indices.append(torch.arange(end - seq_length, start, -seq_length, dtype=torch.int64))

            else:
                indices.append(torch.arange(start, end, seq_length, dtype=torch.int64))

        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySamplerFilter(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len >= flen:
                indices.append(i)

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len >= flen:
                indices.append(i)

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerFilterNrep(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len >= flen:
                repeats = Len;
                indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.NoD = NoD

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len >= flen:
                repeats = Len;
                indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerFilterFix(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len <= flen:
                indices.append(i)
            # if Len>=flen:
            #     indices.append(i)

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len <= flen:
                indices.append(i)
            # if Len>=flen:
            #     indices.append(i)

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerFilterNrepFix(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len <= flen:
                repeats = Len;
                indices += int(repeats) * [i]
            # if Len>=flen:
            #     repeats=Len;
            #     indices+=int(repeats)*[i]

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.NoD = NoD

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        # indices = self.indices[200:201]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len <= flen:
                repeats = Len;
                indices += int(repeats) * [i]
            # if Len>=flen:
            #     repeats=Len;
            #     indices+=int(repeats)*[i]

        indices = torch.tensor(indices)
        self.indices = indices


def len_cal(end_idx, st_path, image_paths):
    with open(st_path, newline='') as f:
        reader = csv.reader(f)
        st_list = list(reader)
    id2st = {name: int(start) for name, start in st_list}
    Lens = []
    for i in range(len(end_idx) - 1):
        init = end_idx[i]
        start = init + id2st[image_paths[init][0].split(os.sep)[-2]]
        Len = end_idx[i + 1] - start
        Lens.append(Len)
    return np.array(Lens)


class MySamplerFilterNrepFixDay5(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, st_path, image_paths):
        indices = []
        self.st_path = st_path
        self.end_idx = end_idx
        self.image_paths = image_paths
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        for i in range(len(end_idx) - 1):
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            Len = end_idx[i + 1] - start
            if Len <= flen:
                repeats = Len;
                indices += int(repeats) * [i]
            # if Len>=flen:
            #     repeats=Len;
            #     indices+=int(repeats)*[i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        # indices = self.indices[200:201]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            Len = self.end_idx[i + 1] - start
            # Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len <= flen:
                repeats = Len
                indices += int(repeats) * [i]
            # if Len>=flen:
            #     repeats=Len;
            #     indices+=int(repeats)*[i]

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerDilate(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, Dil=1, end=96):
        indices = []
        for i in range(len(end_idx) - 1):
            # Len = end_idx[i + 1] - end_idx[i]

            repeats = end / Dil
            indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.end = end

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        # indices = self.indices[200:201]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, Dil):
        indices = []
        for i in range(len(self.end_idx) - 1):
            # Len = self.end_idx[i + 1] - self.end_idx[i]

            repeats = self.end / Dil
            indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerFilterALLFix(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len <= flen:
                chunks = np.array_split(np.arange(0, Len), NoD)
                repeats = 1;
                for chunk in chunks:
                    repeats *= len(chunk)
                indices += repeats * [i]
            # if Len>=flen:
            #     chunks = np.array_split(np.arange(0, Len), NoD)
            #     repeats=1;
            #     for chunk in chunks:
            #         repeats*=len(chunk)
            #     indices+=repeats*[i]

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.NoD = NoD

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len <= flen:
                chunks = np.array_split(np.arange(0, Len), self.NoD)
                repeats = 1;
                for chunk in chunks:
                    repeats *= len(chunk)
                indices += repeats * [i]
            # if Len>=flen:
            #     chunks = np.array_split(np.arange(0, Len), NoD)
            #     repeats=1;
            #     for chunk in chunks:
            #         repeats*=len(chunk)
            #     indices+=repeats*[i]

        indices = torch.tensor(indices)
        self.indices = indices


class MySamplerALLm(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []

        for i in range(len(end_idx[0]) - 1):
            L = []
            for e_i in end_idx:
                L.append(e_i[i + 1] - e_i[i])
            repeats = int(np.min(np.array(L)))
            indices += repeats * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySamplerALLAv(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []

        for i in range(len(end_idx[0]) - 1):
            L = []
            for e_i in end_idx:
                L.append(e_i[i + 1] - e_i[i])
            repeats = int((np.min(np.array(L)) + np.max(np.array(L))) / 2)
            indices += repeats * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySamplerALLBl(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []

        for i in range(len(end_idx[0]) - 1):
            L = []
            for e_i in end_idx:
                L.append(e_i[i + 1] - e_i[i])
            repeats = int(L[-1])
            indices += repeats * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class MySamplerFilterALL(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []
        for i in range(len(end_idx) - 1):

            Len = end_idx[i + 1] - end_idx[i]
            if Len >= flen:
                chunks = np.array_split(np.arange(0, Len), NoD)
                repeats = 1;
                for chunk in chunks:
                    repeats *= len(chunk)
                indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.NoD = NoD

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

    def update(self, flen):
        indices = []
        for i in range(len(self.end_idx) - 1):

            Len = self.end_idx[i + 1] - self.end_idx[i]
            if Len >= flen:
                chunks = np.array_split(np.arange(0, Len), self.NoD)
                repeats = 1;
                for chunk in chunks:
                    repeats *= len(chunk)
                indices += int(repeats) * [i]

        indices = torch.tensor(indices)
        self.indices = indices


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, TriLos=False):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        self.TriLos = TriLos

    def __getitem__(self, index):
        start = index
        binn = np.digitize(index, self.end_idx)
        # size = (400, 410)
        # mask = Image.new('L', (431, 431), 0)
        # draw = ImageDraw.Draw(mask)
        # draw.ellipse((31, 21) + size, fill=255)
        le = self.end_idx[binn] - self.end_idx[binn - 1]
        if (le < self.seq_length):
            end = self.end_idx[binn]
            indices = list(range(end - 1, end)) * self.seq_length
            indices[0:end - start] = list(range(start, end))
        else:
            end = index + self.seq_length
            indices = list(range(start, end))

        images = []
        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)
        if (le < self.seq_length):
            x[end - start:, :, :, :] = 0
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        if self.TriLos == False:
            return x, y
        else:
            return x, y, torch.tensor([binn - 1])

    def __len__(self):
        return self.length


######################################
# class MyDataset(Dataset):
#     def __init__(self, image_paths, seq_length, transform, length, end_idx):
#         self.image_paths = image_paths
#         self.seq_length = seq_length
#         self.transform = transform
#         self.length = length
#         self.end_idx = end_idx
#
#     def __getitem__(self, index):
#         start = index
#         binn = np.digitize(index, self.end_idx)
#         # size = (400, 410)
#         # mask = Image.new('L', (431, 431), 0)
#         # draw = ImageDraw.Draw(mask)
#         # draw.ellipse((31, 21) + size, fill=255)
#         le = self.end_idx[binn] - self.end_idx[binn - 1]
#         if (le < self.seq_length):
#             end = self.end_idx[binn]
#             indices = list(range(end - 1, end)) * self.seq_length
#             # indices = list(range(start, start+1)) * self.seq_length
#
#             indices[0:end - start] = list(range(start, end))
#             # indices[self.seq_length-le:] = list(range(start, end))
#         else:
#             end = index + self.seq_length
#             indices = list(range(start, end))
#
#         images = []
#         seed = np.random.randint(2147483646)
#         for i in indices:
#             #            print(i)
#             image_path = self.image_paths[i][0]
#             image = Image.open(image_path)
#             # ime=ImO.equalize(image)
#             # im3 = ImageChops.multiply(image, mask)
#             # im3.show()
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#             np.random.seed(seed)
#             random.seed(seed)
#
#             torch.backends.cudnn.enabled = False
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#             if self.transform:
#                 image = self.transform(image)
#             #            image=image.repeat(3,1,1)
#             images.append(image)
#         x = torch.stack(images)
#         length=self.seq_length
#         if (le < self.seq_length):
#             x[end - start:, :, :, :] = 102
#             length=le
#             # x[:self.seq_length-le, :, :, :] = 102
#
#         y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
#
#         return x, y, length
#
#     def __len__(self):
#         return self.length

class MyDatasetL(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx

    def __getitem__(self, index):
        start = index
        binn = np.digitize(index, self.end_idx)
        # size = (400, 410)
        # mask = Image.new('L', (431, 431), 0)
        # draw = ImageDraw.Draw(mask)
        # draw.ellipse((31, 21) + size, fill=255)
        le = self.end_idx[binn] - self.end_idx[binn - 1]
        if (le < self.seq_length):
            end = self.end_idx[binn]
            indices = list(range(end - 1, end)) * self.seq_length
            indices[0:end - start] = list(range(start, end))
        else:
            end = index + self.seq_length
            indices = list(range(start, end))

        images = []
        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)
        length = torch.tensor(self.seq_length)

        if (le < self.seq_length):
            x[end - start:, :, :, :] = 0
            length = le
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)

        return x, y, length

    def __len__(self):
        return self.length


class MyDataset1(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, TriLos=False):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        self.TL = TriLos

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # indices=torch.randint(start,end,(self.seq_length,))
        indices = torch.randint(start + self.seq_length - 1, end, (1,))
        # indices = torch.randint(end-1, end, (1,))

        indices = torch.arange(indices[0] - self.seq_length + 1, indices[0] + 1)
        indices, _ = torch.sort(indices, dim=0)
        # indices=torch.tensor([start])
        images = []
        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)

        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index

    def __len__(self):
        return self.length


class MyDataset1end(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, imrange=1, TriLos=False):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        self.TL = TriLos
        self.imrange = imrange

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        initial = max(start, end - self.imrange)
        indices = torch.randint(initial, end, (1,))
        # indices=torch.tensor([end-1])
        images = []
        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)

        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index

    def __len__(self):
        return self.length


class MyDatasetDivide(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=2,
                 cut=1):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            start = self.end_idx[i]
            binn = i + 1
            end = self.end_idx[binn]
            chunks = np.array_split(np.arange(start, end), self.NoD)
            a = np.array(np.meshgrid(*chunks)).T.reshape(-1, len(chunks))
            np.random.shuffle(a)
            self.it.append(a)
        self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # len=math.ceil((end-start)/NoD)
        # chunks = np.array_split(np.arange(start, end), self.NoD)
        # indices=[]
        # for chunk in chunks:
        #     indices.append(chunk[torch.randint(0,len(chunk),(1,))])
        # indices=torch.tensor(indices)
        self.it[index] = np.roll(self.it[index], 1, axis=0)
        indices = torch.tensor(self.it[index][0])
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivideVal(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=2,
                 cut=1):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        # self.it = []
        # for i in range(len(end_idx) - 1):
        #     start = self.end_idx[i]
        #     binn = i + 1
        #     end = self.end_idx[binn]
        #     chunks = np.array_split(np.arange(start, end), self.NoD)
        #     a = np.array(np.meshgrid(*chunks)).T.reshape(-1, 2)
        #     np.random.shuffle(a)
        #     self.it.append(a)
        # self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # len=math.ceil((end-start)/NoD)
        chunks = np.array_split(np.arange(start, end), self.NoD)
        indices = []
        for chunk in chunks:
            indices.append(chunk[-1])
        indices = torch.tensor(indices)
        # self.it[index] = np.roll(self.it[index], 1, axis=0)
        # indices = torch.tensor(self.it[index][0])
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431
        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivide1p(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=2,
                 cut=1):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        # self.it = []
        # for i in range(len(end_idx) - 1):
        #     start = self.end_idx[i]
        #     binn = i + 1
        #     end = self.end_idx[binn]
        #     chunks = np.array_split(np.arange(start, end), self.NoD)
        #     a = np.array(np.meshgrid(*chunks)).T.reshape(-1, 2)
        #     np.random.shuffle(a)
        #     self.it.append(a)
        # self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # len=math.ceil((end-start)/NoD)
        chunks = np.array_split(np.arange(start, end), self.NoD)
        indices = []
        for chunk in chunks:
            indices.append(chunk[torch.randint(0, len(chunk), (1,))])
        indices = torch.tensor(indices)
        # self.it[index] = np.roll(self.it[index], 1, axis=0)
        # indices = torch.tensor(self.it[index][0])
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        # print(self.image_paths[indices[0]][0])
        # print(self.image_paths[indices[0]][0].split("/"))
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431
        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivide1p_Day(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=1,
                 cut=1, end=96):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path

        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut

        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}

        self.st_path = st_path
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        # self.it = []
        # for i in range(len(end_idx) - 1):
        #     start = self.end_idx[i]
        #     binn = i + 1
        #     end = self.end_idx[binn]
        #     chunks = np.array_split(np.arange(start, end), self.NoD)
        #     a = np.array(np.meshgrid(*chunks)).T.reshape(-1, 2)
        #     np.random.shuffle(a)
        #     self.it.append(a)
        # self.test = []

    def __getitem__(self, index):
        init = self.end_idx[index]
        start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
        end = start + self.end
        # binn = index + 1
        # end = self.end_idx[binn]
        # len=math.ceil((end-start)/NoD)
        seq_start = random.randint(start, end)
        # indices = random.sample(chunks,self.seq_length)
        indices = []
        for seq_ind in range(self.seq_length):
            ind = seq_ind + seq_start
            ind = ind - self.end if ind >= end else ind
            indices.append(ind)
        indices = torch.tensor(indices)
        # self.it[index] = np.roll(self.it[index], 1, axis=0)
        # indices = torch.tensor(self.it[index][0])
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        # print(self.image_paths[indices[0]][0])
        # print(self.image_paths[indices[0]][0].split("/"))
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431
        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivide1p_L(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=1,
                 cut=0):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx

        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        duration = end - start
        indices = []
        if duration > self.seq_length:
            # init=torch.randint(start, end, (1,))
            init = 0
        else:

            init = 0
        for ind in range(init, init + self.seq_length):
            if ind >= end:
                indices.append(ind - duration)
            else:
                indices.append(ind)

        indices = torch.tensor(indices)

        images = []

        seed = np.random.randint(2147483646)

        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]

        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431
        for i in indices:

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))

            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                # mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
                mask = ROI
            else:
                mask = image

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)

            images.append(image)

        x = torch.stack(images)

        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index

    def __len__(self):
        return self.length


class MyDatasetALLA(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=2,
                 cut=1):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for end_idx in self.end_idx:
            iit = []
            for i in range(len(end_idx) - 1):
                arr = np.arange(end_idx[i], end_idx[i + 1])
                np.random.shuffle(arr)
                iit.append(arr)
            self.it.append(iit)

    def __getitem__(self, index):
        indices = []
        for i in range(len(self.it)):
            self.it[i][index] = np.roll(self.it[i][index], 1, axis=0)
            indices.append(self.it[i][index][0])
        indices = torch.tensor(indices)
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[0][indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431
        for c, i in enumerate(indices):
            #            print(i)

            image_path = self.image_paths[c][i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.sep)[-3:-1])
                ofpath = os.path.join(a, self.O_path, b, 'imgMed.png')
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
                # mask=ROI[:h,:w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])

        y = torch.tensor([self.image_paths[0][indices[0]][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def updateD(self):
        self.it = []
        for end_idx in self.end_idx:
            iit = []
            for i in range(len(end_idx) - 1):
                arr = np.arange(end_idx[i], end_idx[i + 1])
                np.random.shuffle(arr)
                iit.append(arr)
            self.it.append(iit)

    def __len__(self):
        return self.length


class MyDatasetDivideNrep(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt, NoD=2,
                 cut=1):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            start = self.end_idx[i]
            binn = i + 1
            end = self.end_idx[binn]
            chunks = np.array_split(np.arange(start, end), self.NoD)
            # lens=[]
            for ii in range(len(chunks)):
                np.random.shuffle(chunks[ii])
                # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))
            Mi = 0
            chunkM = chunks[Mi]
            chunkO = [x for ii, x in enumerate(chunks) if ii != Mi]
            cor = []
            for ii in chunkM:
                lc = [ii]
                for c in range(len(chunkO)):
                    chunkO[c] = np.roll(chunkO[c], 1, axis=0)
                    lc.append(chunkO[c][0])
                cor.append(lc)

            self.it.append(np.array(cor))
        self.test = []

    def updateD(self):
        self.it = []
        for i in range(len(self.end_idx) - 1):
            start = self.end_idx[i]
            binn = i + 1
            end = self.end_idx[binn]
            chunks = np.array_split(np.arange(start, end), self.NoD)
            # lens=[]
            for ii in range(len(chunks)):
                np.random.shuffle(chunks[ii])
                # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))
            Mi = 0
            chunkM = chunks[Mi]
            chunkO = [x for ii, x in enumerate(chunks) if ii != Mi]
            cor = []
            for ii in chunkM:
                lc = [ii]
                for c in range(len(chunkO)):
                    chunkO[c] = np.roll(chunkO[c], 1, axis=0)
                    if len(chunkO[c]) > 0:
                        lc.append(chunkO[c][0])
                cor.append(lc)

            self.it.append(np.array(cor))
        self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # len=math.ceil((end-start)/NoD)
        # chunks = np.array_split(np.arange(start, end), self.NoD)
        # indices=[]
        # for chunk in chunks:
        #     indices.append(chunk[torch.randint(0,len(chunk),(1,))])
        # indices=torch.tensor(indices)
        self.it[index] = np.roll(self.it[index], 1, axis=0)
        indices = torch.tensor(self.it[index][0])
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h = image.shape[0] - y - 1
                if x + w > image.shape[1]:
                    w = image.shape[1] - x - 1
                ROI = image[y:y + h, x:x + w]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h > mask.shape[0]:
                    h = mask.shape[0] - y1 - 1
                if x1 + w > mask.shape[1]:
                    w = mask.shape[1] - x1 - 1
                mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivideNrep_Day(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=2,
                 cut=1, end=96):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        self.st_path = st_path
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            # start = self.end_idx[i]
            # binn = i + 1
            # end = self.end_idx[binn]
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = start + self.end
            chunks = np.arange(start, end)
            # chunks=np.array([start,start,start])

            # lens=[]
            np.random.shuffle(chunks)
            # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))

            self.it.append(chunks)
        self.test = []

    def updateD(self):
        self.it = []
        for i in range(len(self.end_idx) - 1):
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = start + self.end
            chunks = np.arange(start, end)
            np.random.shuffle(chunks)
            self.it.append(chunks)
        self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]
        indices = torch.tensor([self.it[index][0]])
        self.it[index] = np.roll(self.it[index], -1, axis=0)

        # indices = self.it[index]
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h_n = image.shape[0] - y - 1
                else:
                    h_n = h
                if x + w > image.shape[1]:
                    w_n = image.shape[1] - x - 1
                else:
                    w_n = w
                ROI = image[y:y + h_n, x:x + w_n]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h_n > mask.shape[0]:
                    h_n = mask.shape[0] - y1 - 1
                else:
                    h_n = h_n
                if x1 + w_n > mask.shape[1]:
                    w_n = mask.shape[1] - x1 - 1
                else:
                    w_n = w_n
                mask[y1:y1 + h_n, x1:x1 + w_n] = ROI[:h_n, :w_n]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetDivideNrep_Day_tsc(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=2,
                 cut=1, end=96, dil=1, day5=0):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        self.st_path = st_path
        self.day5 = day5
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):

            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            if self.day5:
                end = min(self.end_idx[i + 1], start + self.end)
            else:
                end = start + self.end
            chunks = np.arange(start, end, step=dil)
            self.it.append(chunks)
        self.test = []
        self.dil = dil

    def updateD(self):
        self.it = []
        for i in range(len(self.end_idx) - 1):

            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            if self.day5:
                end = min(self.end_idx[i + 1], start + self.end)
            else:
                end = start + self.end
            chunks = np.arange(start, end, step=self.dil)
            self.it.append(chunks)
        self.test = []

    def __getitem__(self, index):

        indices = torch.tensor(self.it[index])
        start = self.it[index][0]
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lensT = len(self.it[index])
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h_n = image.shape[0] - y - 1
                else:
                    h_n = h
                if x + w > image.shape[1]:
                    w_n = image.shape[1] - x - 1
                else:
                    w_n = w
                ROI = image[y:y + h_n, x:x + w_n]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h_n > mask.shape[0]:
                    h_n = mask.shape[0] - y1 - 1
                else:
                    h_n = h_n
                if x1 + w_n > mask.shape[1]:
                    w_n = mask.shape[1] - x1 - 1
                else:
                    w_n = w_n
                mask[y1:y1 + h_n, x1:x1 + w_n] = ROI[:h_n, :w_n]
            else:
                mask = image

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)

            images.append(image)

        x = torch.stack(images)
        if self.day5:
            x = torch.nn.functional.pad(x.transpose(0, 3), (0, self.end // self.dil - x.shape[0])).transpose(0, 3)

        o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o, id

    def __len__(self):
        return self.length

    def __len__(self):
        return self.length


import h5py


class MyDataset_pd_tsc(Dataset):
    def __init__(self, data_path, dil=1, day5=0, end=96):

        self.day5 = day5
        self.dil = dil
        self.end = end

        with h5py.File(data_path, "r") as f:
            # List all groups
            # print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]

            self.data = []
            for cc, key in enumerate(list(f.keys())):
                # Get the data
                self.data.append(list(f[key]))
        self.length = len(self.data[0])

    def __getitem__(self, index):

        id = self.data[1][index]
        label = self.data[2][index]
        features = self.data[0][index]
        out = self.data[4][index]
        lenST = self.data[3][index]

        y = torch.tensor(label)
        out = torch.tensor(out)
        BS = features.shape
        if self.day5:
            features[lenST:, :] = 0
            # out[lenST:]=0
        # out=torch.mean(out)
        x = torch.from_numpy(features).contiguous().view(-1, self.dil, BS[-1])[:, 0, :].squeeze(1)
        return x, y, index, lenST, out, id

    def __len__(self):
        return self.length

    def __len__(self):
        return self.length


class MyDatasetDivideNrep_Day5(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, st_path, crpc, opt,
                 NoD=2,
                 cut=1, end=96):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        self.NoD = NoD
        self.cut = cut
        self.st_path = st_path
        with open(st_path, newline='') as f:
            reader = csv.reader(f)
            self.st_list = list(reader)
        self.id2st = {name: int(start) for name, start in self.st_list}
        self.end = end
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}
        self.it = []
        for i in range(len(end_idx) - 1):
            # start = self.end_idx[i]
            # binn = i + 1
            # end = self.end_idx[binn]
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = self.end_idx[i + 1]
            chunks = np.arange(start, end)
            # chunks=np.array([start,start,start])
            # lens=[]
            np.random.shuffle(chunks)
            # lens.append(len(chunks[ii]))
            # Mi=lens.index(max(lens))

            self.it.append(chunks)
        self.test = []

    def updateD(self):
        self.it = []
        for i in range(len(self.end_idx) - 1):
            init = self.end_idx[i]
            start = init + self.id2st[self.image_paths[init][0].split(os.sep)[-2]]
            end = self.end_idx[i + 1]
            chunks = np.arange(start, end)
            np.random.shuffle(chunks)
            self.it.append(chunks)
        self.test = []

    def __getitem__(self, index):
        start = self.end_idx[index]

        indices = torch.tensor([self.it[index][0]])
        # indices = self.it[index]
        self.it[index] = np.roll(self.it[index], -1, axis=0)
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.cut:
            if self.C_dict[id]:

                x, y, w, h = self.C_dict[id][-1]
                x -= 10
                y -= 10
                w += 10
                h += 10
            else:
                x, y, w, h = 0, 0, 431, 431

        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=10))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.cut:
                if self.cr == 1:

                    mask = np.ones(image.shape, dtype=np.uint8) * 102
                else:
                    mask = np.zeros(image.shape, dtype=np.uint8)
                ROI = np.zeros((h, w))
                if y + h > image.shape[0]:
                    h_n = image.shape[0] - y - 1
                else:
                    h_n = h
                if x + w > image.shape[1]:
                    w_n = image.shape[1] - x - 1
                else:
                    w_n = w
                ROI = image[y:y + h_n, x:x + w_n]
                x1 = image.shape[0] // 2 - ROI.shape[0] // 2
                y1 = image.shape[1] // 2 - ROI.shape[1] // 2
                if y1 + h_n > mask.shape[0]:
                    h_n = mask.shape[0] - y1 - 1
                else:
                    h_n = h_n
                if x1 + w_n > mask.shape[1]:
                    w_n = mask.shape[1] - x1 - 1
                else:
                    w_n = w_n
                mask[y1:y1 + h_n, x1:x1 + w_n] = ROI[:h_n, :w_n]
            else:
                mask = image
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDatasetWhole(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx, L_path, C_path, O_path, crpc, opt):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        # self.L_path=L_path
        # self.C_path=C_path
        self.O_path = O_path
        self.crpc = Crp(c=crpc)
        self.cr = crpc
        self.opt = opt
        with open(C_path, 'rb') as handle:
            self.C_dict = pickle.load(handle)
        with open(L_path, newline='') as f:
            reader = csv.reader(f)
            self.Len_list = list(reader)
        self.id2len = {z: [L1, L2, L4, L5, L6] for z, L1, L2, L3, L4, L5, L6 in self.Len_list}

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        indices = torch.arange(start, end)
        images = []
        ofimages = []
        seed = np.random.randint(2147483646)
        id = self.image_paths[indices[0]][0].split(os.path.sep)[-2]
        lens = self.id2len[id]
        lensT = torch.tensor(list(map(float, lens)))
        if self.C_dict[id]:

            x, y, w, h = self.C_dict[id][-1]
            x -= 10
            y -= 10
            w += 10
            h += 10
        else:
            x, y, w, h = 0, 0, 431, 431
        for i in indices:
            #            print(i)

            image_path = self.image_paths[i][0]
            image = np.array(self.crpc(Image.open(image_path)))
            if self.opt:
                a = Path(image_path).parents[4]
                b = os.path.join(*image_path.split(os.path.sep)[-4:-1])
                ofpath = os.path.join(a, self.O_path, b, 'img{}.png'.format(i - start))
                ofimg = Image.open(ofpath)
                ofimg = ofimg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.UnsharpMask(radius=15)).filter(
                    ImageFilter.GaussianBlur(radius=5))
                ofimg = np.array(ofimg)
                s1 = ofimg.shape[0] / 2 - (y + h / 2)
                s2 = ofimg.shape[1] / 2 - (x + w / 2)
                ofimg = np.roll(ofimg, int(s1), axis=0)
                ofimg = np.roll(ofimg, int(s2), axis=1)
                ofimg = Image.fromarray(ofimg)
            if self.crpc == 1:

                mask = np.ones(image.shape, dtype=np.uint8) * 102
            else:
                mask = np.zeros(image.shape, dtype=np.uint8)
            ROI = np.zeros((h, w))
            if y + h > image.shape[0]:
                h = image.shape[0] - y - 1
            if x + w > image.shape[1]:
                w = image.shape[1] - x - 1
            ROI = image[y:y + h, x:x + w]
            x1 = image.shape[0] // 2 - ROI.shape[0] // 2
            y1 = image.shape[1] // 2 - ROI.shape[1] // 2
            if y1 + h > mask.shape[0]:
                h = mask.shape[0] - y1 - 1
            if x1 + w > mask.shape[1]:
                w = mask.shape[1] - x1 - 1
            mask[y1:y1 + h, x1:x1 + w] = ROI[:h, :w]
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            image = Image.fromarray(mask)
            if self.transform:
                image = self.transform(image)
                if self.opt:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    ofimg = self.transform(ofimg)
            #            image=image.repeat(3,1,1)
            images.append(image)
            if self.opt:
                ofimages.append(ofimg)
            # if self.opt:

        x = torch.stack(images)
        if self.opt:
            o = torch.stack(ofimages)
        else:
            o = torch.tensor([])
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y, index, lensT, o

    def __len__(self):
        return self.length


class MyDataset2(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        self.it = []
        for i in range(length):
            # self.it.append(torch.arange(end_idx[i],end_idx[i+1]))
            self.it.append(
                torch.tensor(random.sample(range(end_idx[i], end_idx[i + 1]), int(end_idx[i + 1] - end_idx[i]))))

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # indices=torch.randint(start,end,(1,))
        # indices=torch.tensor([next(self.it[index])])
        self.it[index] = roll(self.it[index], 1, 0)
        indices = torch.tensor([self.it[index][0]])
        images = []

        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)

        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.length


class MyDataset2TL(Dataset):
    def __init__(self, image_paths, seq_length, transform, length, end_idx):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.end_idx = end_idx
        self.it = []
        for i in range(length):
            # self.it.append(torch.arange(end_idx[i],end_idx[i+1]))
            self.it.append(torch.arange(end_idx[i], end_idx[i + 1] + 1))
            self.it[-1][-1] = 0

    def __getitem__(self, index):
        start = self.end_idx[index]
        binn = index + 1
        end = self.end_idx[binn]
        # indices=torch.randint(start,end,(1,))
        # indices=torch.tensor([next(self.it[index])])
        self.it[index][-1] += 1
        indices = torch.tensor([self.it[index][self.it[index][-1] - 1], self.it[index][-self.it[index][-1] - 1]])
        if self.it[index][-1] == self.it[index].size()[0] - 1:
            self.it[index][-1] = 0
        images = []

        seed = np.random.randint(2147483646)
        for i in indices:
            #            print(i)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)
        x = torch.stack(images)

        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        return x, torch.tensor([[y], [y]]), torch.tensor([[index], [index]])

    def __len__(self):
        return self.length


class MySamplerALL(torch.utils.data.sampler.Sampler):
    def __init__(self, end_idx, flen, NoD=2):
        indices = []
        for i in range(len(end_idx[0]) - 1):
            L = []

            for e_i in end_idx:
                L.append(e_i[i + 1] - e_i[i])
            repeats = int(np.max(np.array(L)))
            indices += repeats * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


def datPP3(root_dir, i):
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    #    class_image_paths=[]
    c_p1 = []
    c_p2 = []
    c_p3 = []

    #    end_idx = []
    e_i1 = []
    e_i2 = []
    e_i3 = []
    ratio = [0.20, 0.20, 0.20, 0.20, 0.20]
    list_d = []
    for c, class_path in enumerate(class_paths):
        list_d = []
        for d in os.scandir(class_path):
            if d.is_dir:
                list_d.append(d)
        ld = len(list_d)
        lengths = [int(ld * ratio[0]), int(ld * ratio[1]), int(ld * ratio[2]), int(ld * ratio[3]), int(ld * ratio[4])]
        if (sum(lengths) != ld):
            # lengths[0] += ld - sum(lengths)
            for ic in range(ld - sum(lengths)):
                lengths[ic] += 1
        random.seed(13722)
        shuffle(list_d)
        Inputt = iter(list_d)
        S_d1 = [list(islice(Inputt, elem))
                for elem in lengths]
        # for cu, el in enumerate(S_d1):
        #     with open('S_D_{}.txt'.format(cu), 'a') as f:
        #         for ele in el:
        #             f.write(str(ele.name) + '\n')
        S_d = list([[], []])
        x = [x for x in range(5) if x != i]
        for ii in x:
            S_d[0].extend(S_d1[ii])
        S_d[1] = S_d1[i]
        for d in S_d[0]:
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                paths = [(p, c) for p in paths]
                c_p1.extend(paths)
                e_i1.extend([len(paths)])
        for d in S_d[1]:
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                paths = [(p, c) for p in paths]
                c_p2.extend(paths)
                e_i2.extend([len(paths)])
        print(torch.sum(torch.tensor(e_i2)))
    #        for d in os.scandir(class_path):
    #            if d.is_dir:
    #                paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
    #                # Add class idx to paths
    #                paths = [(p, c) for p in paths]
    #                class_image_paths.extend(paths)
    #                end_idx.extend([len(paths)])
    #
    #    end_idx = [0, *end_idx]
    #    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    e_i1 = [0, *e_i1]
    e_i2 = [0, *e_i2]

    e_i1 = torch.cumsum(torch.tensor(e_i1), 0)
    e_i2 = torch.cumsum(torch.tensor(e_i2), 0)

    #    return class_image_paths,end_idx
    return c_p1, c_p2, e_i1, e_i2


def datPP4(root_dir, i=2, i2=4):
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    #    class_image_paths=[]
    c_p1 = []
    c_p2 = []
    c_p3 = []

    #    end_idx = []
    e_i1 = []
    e_i2 = []
    e_i3 = []
    ratio = [0.23, 0.23, 0.15, 0.19, 0.20]
    list_d = []
    for c, class_path in enumerate(class_paths):
        list_d = []
        for d in os.scandir(class_path):
            if d.is_dir:
                list_d.append(d)
        ld = len(list_d)
        lengths = [int(ld * ratio[0]), int(ld * ratio[1]), int(ld * ratio[2]), int(ld * ratio[3]), int(ld * ratio[4])]
        if (sum(lengths) != ld):
            # lengths[0] += ld - sum(lengths)
            for i in range(ld - sum(lengths)):
                lengths[i] += 1
        random.seed(1)
        shuffle(list_d)
        Inputt = iter(list_d)
        S_d1 = [list(islice(Inputt, elem))
                for elem in lengths]
        S_d = list([[], [], []])
        x = [x for x in range(5) if ((x != i) and (x != i2))]
        for ii in x:
            S_d[0].extend(S_d1[ii])
        S_d[2] = S_d1[i]
        S_d[1] = S_d1[i2]
        for d in S_d[0]:
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                paths = [(p, c) for p in paths]
                c_p1.extend(paths)
                e_i1.extend([len(paths)])
        for d in S_d[1]:
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                paths = [(p, c) for p in paths]
                c_p2.extend(paths)
                e_i2.extend([len(paths)])
        for d in S_d[2]:
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                paths = [(p, c) for p in paths]
                c_p3.extend(paths)
                e_i3.extend([len(paths)])

    #        for d in os.scandir(class_path):
    #            if d.is_dir:
    #                paths = sorted(glob.glob(os.path.join(d.path, '*.jpg')))
    #                # Add class idx to paths
    #                paths = [(p, c) for p in paths]
    #                class_image_paths.extend(paths)
    #                end_idx.extend([len(paths)])
    #
    #    end_idx = [0, *end_idx]
    #    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
    e_i1 = [0, *e_i1]
    e_i2 = [0, *e_i2]
    e_i3 = [0, *e_i3]

    e_i1 = torch.cumsum(torch.tensor(e_i1), 0)
    e_i2 = torch.cumsum(torch.tensor(e_i2), 0)
    e_i3 = torch.cumsum(torch.tensor(e_i3), 0)

    #    return class_image_paths,end_idx
    return c_p1, c_p2, c_p3, e_i1, e_i2, e_i3
