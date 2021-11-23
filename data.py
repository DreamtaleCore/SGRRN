""" data.py

data I/O for low-level vision tasks, ie, reflection removal

@autor: DreamTale
"""

import os.path
import random
import torch
import cv2
import numpy as np
import scipy.stats as st
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

# Image utils
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(s_dir):
    images = []
    assert os.path.isdir(s_dir), '%s is not a valid directory' % s_dir

    for root, _, fnames in sorted(os.walk(s_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 2] = s / 3.0
    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    return irg


def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg


def label2colormap(label):
    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
    cmap[:, :, 2] = (m & 4) << 5
    return cmap


def cvt_dataset_to_class(data_list):
    """
    Convert the list of data into a standard classify task form
    :param data_list: [dataset, dataset, ...]
    :return: xs, ys
    """
    xs = []
    ys = []
    for ii, elem in enumerate(data_list):
        bath_size = elem.shape[0]
        for jj in range(bath_size):
            xs += [elem[jj].unsqueeze(0)]
        ys += [ii] * bath_size

    magic_num = np.random.randint(1, 2**31)

    random.seed(magic_num)
    random.shuffle(xs)
    random.seed(magic_num)
    random.shuffle(ys)

    xs = Variable(torch.cat(xs, dim=0))
    ys = Variable(torch.tensor(ys))

    return xs, ys


class LowLevelImageFolder(data.Dataset):
    """
    A loader for loading most of low level tasks
    """

    def __init__(self, root, sub_folders, new_size=350, mode='semantic-rr-syn', img_loader=cv2.imread):
        image_list_dict = {}
        for sub_folder in sub_folders:
            sub_root = os.path.join(root, sub_folder)
            if os.path.isdir(sub_root):
                image_list_dict[sub_folder] = make_dataset(sub_root)

        self.image_list_dict = image_list_dict
        self.root = root
        self.mode = mode
        self.new_size = new_size
        self.img_loader = img_loader

        # create a vignetting mask
        self.additional_attributes = []

    def gkern(self, kernel_len=100, n_sig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * n_sig + 1.) / kernel_len
        x = np.linspace(-n_sig - interval / 2., n_sig + interval / 2., kernel_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def data_argument(self, img, new_w, new_h, random_flip=0, random_pos=None, resize_mode=None, is_pad=False):

        # img = rotate(img,random_angle, order = mode)
        h, w = img.shape[:2]
        random_pos[1] = random_pos[1] if random_pos[1] < h - 1 else h - 1
        random_pos[3] = random_pos[3] if random_pos[3] < w - 1 else w - 1
        if random_pos is not None:
            if len(img.shape) > 2:
                img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
            else:
                img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3]]
        if resize_mode is not None:
            img = cv2.resize(img, (new_w, new_h), interpolation=resize_mode)
        else:
            img = cv2.resize(img, (new_w, new_h))
        if is_pad:
            img = cv2.resize(img, (new_w, new_h), interpolation=resize_mode)
            max_size = max(new_w, new_h)
            if len(img.shape) > 2:
                new_img = np.zeros((max_size, max_size, 3))
                new_img[:new_h, :new_w, :] = img
            else:
                new_img = np.zeros((max_size, max_size))
                new_img[:new_h, :new_w] = img
            img = new_img

        if random_flip > 0.5:
            img = np.fliplr(img)

        return img.copy()

    def syn_image(self, t, r, sigma):
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0

        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[0: 2]
        # print(r_blur.shape)
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[newh:newh + h, neww:neww + w, :]
        alpha2 = 1 - np.random.random() / 5.0
        r_blur_mask = np.multiply(r_blur, alpha1)
        blend = r_blur_mask + t * alpha2

        t = np.power(t, 1 / 2.2)
        r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        return t, r_blur_mask, blend

    def __len__(self):
        if 'semantic-rr-syn' in self.mode:
            return 10000
        min_len = 2 ** 31
        for val in self.image_list_dict.values():
            min_len = len(val) if len(val) < min_len else min_len
        return min_len - 1

    def remove_additional_attributes(self):
        """
        Call this function whenever the generator meets end
        :return:
        """
        for attri in self.additional_attributes:
            try:
                delattr(self, attri)
            except Exception as e:
                pass

    def get_data(self, index):
        if self.mode == 'syn-reflection':
            # transmission_layer, reflection_layer
            assert len(self.image_list_dict.keys()) == 2

            if not hasattr(self, 'transmission_list'):
                ids = np.random.permutation(self.__len__())
                # for synthetic images
                self.additional_attributes.append('transmission_list')
                self.additional_attributes.append('reflection_list')
                self.additional_attributes.append('k_sz')
                self.additional_attributes.append('g_mask')
                self.k_sz = np.linspace(1, 5, 80)
                g_mask = self.gkern(560, 3)
                self.g_mask = np.dstack((g_mask, g_mask, g_mask))

                for key in self.image_list_dict.keys():
                    if 'reflection' in key:
                        self.reflection_list = list(np.array(self.image_list_dict[key])[list(ids)])
                    else:
                        self.transmission_list = list(np.array(self.image_list_dict[key])[list(ids)])

            syn_image1 = cv2.imread(self.transmission_list[index], -1)[:, :, ::-1]
            syn_image2 = cv2.imread(self.reflection_list[index], -1)[:, :, ::-1]

            new_w = np.random.randint(256, 480)
            new_h = round((new_w / syn_image1.shape[1]) * syn_image1.shape[0])
            if new_h > 500:
                new_w = int(new_w / new_h * 500)
                new_h = 500

            output_image_t = cv2.resize(np.float32(syn_image1), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_r = cv2.resize(np.float32(syn_image2), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            file_t = os.path.splitext(os.path.basename(self.transmission_list[index]))[0]
            file_r = os.path.splitext(os.path.basename(self.reflection_list[index]))[0]
            file = '{}+{}'.format(file_t, file_r)
            sigma = self.k_sz[np.random.randint(0, len(self.k_sz))]
            if np.mean(output_image_t) * 1 / 2 > np.mean(output_image_r):
                ret_flag = False
            else:
                ret_flag = True
            _, output_image_r, input_image = self.syn_image(output_image_t, output_image_r, sigma)

            return input_image, output_image_t, output_image_r, file, ret_flag
        elif self.mode == 'real-reflection-wo_r':
            # blended, transmission_layer, file name must consistent
            assert len(self.image_list_dict.keys()) == 2

            if not hasattr(self, 'transmission_list'):
                ids = np.random.permutation(self.__len__())
                # for synthetic images
                self.additional_attributes.append('transmission_list')
                self.additional_attributes.append('input_list')

                for key in self.image_list_dict.keys():
                    if 'blended' in key or 'input' in key:
                        self.input_list = list(np.array(self.image_list_dict[key])[list(ids)])
                    else:
                        self.transmission_list = list(np.array(self.image_list_dict[key])[list(ids)])

            try:
                img_in = cv2.imread(self.input_list[index], -1)[:, :, ::-1]
                img_bg = cv2.imread(self.transmission_list[index], -1)[:, :, ::-1]

                file = os.path.splitext(os.path.basename(self.input_list[index]))[0]
                ret_flag = True
            except Exception:
                img_in = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                img_bg = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)

                file = ''
                ret_flag = False

            new_w = np.random.randint(256, 480)
            new_h = round((new_w / img_in.shape[1]) * img_in.shape[0])

            input_image = cv2.resize(np.float32(img_in), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_t = cv2.resize(np.float32(img_bg), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_r = None

            # Do normal DA
            random_flip = random.random()
            random_start_y = random.randint(0, 19)
            random_start_x = random.randint(0, 19)
            random_pos = [random_start_y, random_start_y + input_image.shape[0] - 20, random_start_x,
                          random_start_x + input_image.shape[1] - 20]
            input_image = self.data_argument(input_image, new_w=new_w, new_h=new_h,
                                             random_flip=random_flip, random_pos=random_pos, resize_mode=1)
            output_image_t = self.data_argument(output_image_t, new_w=new_w, new_h=new_h,
                                                random_flip=random_flip, random_pos=random_pos, resize_mode=1)

            return input_image, output_image_t, output_image_r, file, ret_flag
        elif self.mode == 'real-reflection':
            # blended, transmission_layer, reflection_layer, file name must consistent
            assert len(self.image_list_dict.keys()) == 3

            if not hasattr(self, 'transmission_list'):
                ids = np.random.permutation(self.__len__())
                # for synthetic images
                self.additional_attributes.append('transmission_list')
                self.additional_attributes.append('reflection_list')
                self.additional_attributes.append('input_list')

                for key in self.image_list_dict.keys():
                    if 'reflection' in key:
                        self.reflection_list = list(np.array(self.image_list_dict[key])[list(ids)])
                    elif 'blended' in key or 'input' in key:
                        self.input_list = list(np.array(self.image_list_dict[key])[list(ids)])
                    else:
                        self.transmission_list = list(np.array(self.image_list_dict[key])[list(ids)])

            try:
                img_in = cv2.imread(self.input_list[index], -1)[:, :, ::-1]
                img_bg = cv2.imread(self.transmission_list[index], -1)[:, :, ::-1]
                img_rf = cv2.imread(self.reflection_list[index], -1)[:, :, ::-1]

                file = os.path.splitext(os.path.basename(self.input_list[index]))[0]
                ret_flag = True
            except Exception:
                img_in = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                img_bg = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                img_rf = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)

                file = ''
                ret_flag = False

            new_w = np.random.randint(256, 480)
            new_h = round((new_w / img_in.shape[1]) * img_in.shape[0])

            input_image = cv2.resize(np.float32(img_in), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_t = cv2.resize(np.float32(img_bg), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_r = cv2.resize(np.float32(img_rf), (new_w, new_h), cv2.INTER_CUBIC) / 255.0

            # Do normal DA
            random_flip = random.random()
            random_start_y = random.randint(0, 19)
            random_start_x = random.randint(0, 19)
            random_pos = [random_start_y, random_start_y + input_image.shape[0] - 20, random_start_x,
                          random_start_x + input_image.shape[1] - 20]
            input_image = self.data_argument(input_image, new_w=new_w, new_h=new_h,
                                             random_flip=random_flip, random_pos=random_pos,
                                             resize_mode=cv2.INTER_CUBIC)
            output_image_t = self.data_argument(output_image_t, new_w=new_w, new_h=new_h,
                                                random_flip=random_flip, random_pos=random_pos,
                                                resize_mode=cv2.INTER_CUBIC)
            output_image_r = self.data_argument(output_image_r, new_w=new_w, new_h=new_h,
                                                random_flip=random_flip, random_pos=random_pos,
                                                resize_mode=cv2.INTER_CUBIC)

            return input_image, output_image_t, output_image_r, file, ret_flag
        elif self.mode == 'semantic-rr-syn':
            if not hasattr(self, 'voc_dataset'):
                self.additional_attributes.append('voc_dataset')
                self.additional_attributes.append('len_dataset')
                self.additional_attributes.append('len_dataset')
                self.voc_dataset = VOCDataset(self.root)
                self.len_dataset = len(self.voc_dataset)

                self.additional_attributes.append('bg_ids')
                self.additional_attributes.append('rf_ids')
                self.bg_ids = np.random.permutation(self.len_dataset)
                self.rf_ids = np.random.permutation(self.len_dataset)
                # for synthetic images
                self.additional_attributes.append('transmission_list')
                self.additional_attributes.append('reflection_list')
                self.additional_attributes.append('k_sz')
                self.additional_attributes.append('g_mask')
                self.k_sz = np.linspace(1, 5, 80)
                g_mask = self.gkern(560, 3)
                self.g_mask = np.dstack((g_mask, g_mask, g_mask))

            bg_id = self.bg_ids[index % self.len_dataset]
            rf_id = self.rf_ids[index % self.len_dataset]

            bg_dict = self.voc_dataset[bg_id]
            rf_dict = self.voc_dataset[rf_id]

            syn_image1 = bg_dict['image']
            syn_image2 = rf_dict['image']
            syn_seman1 = bg_dict['segmentation']
            syn_seman2 = rf_dict['segmentation']
            syn_name1 = bg_dict['name']
            syn_name2 = rf_dict['name']
            file = '{}+{}'.format(syn_name1, syn_name2)

            new_w = np.random.randint(256, 400)
            new_h = round((new_w / syn_image1.shape[1]) * syn_image1.shape[0])
            if new_h > 400:
                new_w = int(new_w / new_h * 400)
                new_h = 400

            output_image_t = cv2.resize(np.float32(syn_image1), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_image_r = cv2.resize(np.float32(syn_image2), (new_w, new_h), cv2.INTER_CUBIC) / 255.0
            output_seman_t = cv2.resize(syn_seman1, (new_w, new_h), cv2.INTER_NEAREST)
            output_seman_r = cv2.resize(syn_seman2, (new_w, new_h), cv2.INTER_NEAREST)

            sigma = self.k_sz[np.random.randint(0, len(self.k_sz))]
            if np.mean(output_image_t) * 1 / 2 > np.mean(output_image_r):
                ret_flag = False
            else:
                ret_flag = True
            _, output_image_r, input_image = self.syn_image(output_image_t, output_image_r, sigma)

            h, w = input_image.shape[:2]
            if h > w:
                new_w = int(w / h * self.new_size)
                new_h = self.new_size
            else:
                new_h = int(h / w * self.new_size)
                new_w = self.new_size

            input_image = self.data_argument(input_image, new_h=new_h, new_w=new_w, is_pad=True)
            output_image_t = self.data_argument(output_image_t, new_h=new_h, new_w=new_w, is_pad=True)
            output_image_r = self.data_argument(output_image_r, new_h=new_h, new_w=new_w, is_pad=True)
            output_seman_t = self.data_argument(output_seman_t, new_h=new_h, new_w=new_w, is_pad=True,
                                                resize_mode=cv2.INTER_NEAREST)
            output_seman_r = self.data_argument(output_seman_r, new_h=new_h, new_w=new_w, is_pad=True,
                                                resize_mode=cv2.INTER_NEAREST)

            return input_image, output_image_t, output_image_r, output_seman_t, output_seman_r, file, ret_flag
        elif self.mode == 'semantic-rr':
            # file names in [input, background, reflection, bg-semantic, rf-semantic] must consistent
            assert len(self.image_list_dict.keys()) == 5
            if not hasattr(self, 'background_list'):
                ids = np.random.permutation(self.__len__())
                self.additional_attributes.append('background_list')
                self.additional_attributes.append('reflection_list')
                self.additional_attributes.append('semantic_b_list')
                self.additional_attributes.append('semantic_r_list')
                self.additional_attributes.append('input_list')

                self.input_list = list(np.array(self.image_list_dict['input'])[list(ids)])
                self.background_list = list(np.array(self.image_list_dict['background'])[list(ids)])
                self.reflection_list = list(np.array(self.image_list_dict['reflection'])[list(ids)])
                self.semantic_b_list = list(np.array(self.image_list_dict['bg-semantic'])[list(ids)])
                self.semantic_r_list = list(np.array(self.image_list_dict['rf-semantic'])[list(ids)])

            try:
                img_in = cv2.imread(self.input_list[index])[:, :, ::-1]
                img_bg = cv2.imread(self.background_list[index])[:, :, ::-1]
                img_rf = cv2.imread(self.reflection_list[index])[:, :, ::-1]
                sem_bg = cv2.imread(self.semantic_b_list[index], -1)
                sem_rf = cv2.imread(self.semantic_r_list[index], -1)

                file = os.path.splitext(os.path.basename(self.input_list[index]))[0]
                ret_flag = True
            except Exception:
                img_in = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                img_bg = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                img_rf = np.zeros((self.new_size, self.new_size, 3), dtype=np.uint8)
                sem_bg = np.zeros((self.new_size, self.new_size), dtype=np.uint8)
                sem_rf = np.zeros((self.new_size, self.new_size), dtype=np.uint8)

                file = ''
                ret_flag = False

            new_w = self.new_size
            new_h = round((new_w / img_in.shape[1]) * img_in.shape[0])

            sem_bg[sem_bg > 20] = 0
            sem_rf[sem_rf > 20] = 0
            sem_bg[sem_bg < 0] = 0
            sem_rf[sem_rf < 0] = 0

            input_image = cv2.resize(np.float32(img_in), (new_w, new_h)) / 255.0
            output_image_t = cv2.resize(np.float32(img_bg), (new_w, new_h)) / 255.0
            output_image_r = cv2.resize(np.float32(img_rf), (new_w, new_h)) / 255.0
            output_seman_t = cv2.resize(np.uint8(sem_bg), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            output_seman_r = cv2.resize(np.uint8(sem_rf), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Do normal DA
            random_flip = random.random()
            random_start_y = random.randint(0, 19)
            random_start_x = random.randint(0, 19)
            random_pos = [random_start_y, random_start_y + input_image.shape[0] - 20, random_start_x,
                          random_start_x + input_image.shape[1] - 20]
            input_image = self.data_argument(input_image, new_w=self.new_size, new_h=self.new_size,
                                             random_flip=random_flip, random_pos=random_pos)
            output_image_t = self.data_argument(output_image_t, new_w=self.new_size, new_h=self.new_size,
                                                random_flip=random_flip, random_pos=random_pos)
            output_image_r = self.data_argument(output_image_r, new_w=self.new_size, new_h=self.new_size,
                                                random_flip=random_flip, random_pos=random_pos)
            output_seman_t = self.data_argument(output_seman_t, new_w=self.new_size, new_h=self.new_size,
                                                random_flip=random_flip, random_pos=random_pos,
                                                resize_mode=cv2.INTER_NEAREST)
            output_seman_r = self.data_argument(output_seman_r, new_w=self.new_size, new_h=self.new_size,
                                                random_flip=random_flip, random_pos=random_pos,
                                                resize_mode=cv2.INTER_NEAREST)

            return input_image, output_image_t, output_image_r, output_seman_t, output_seman_r, file, ret_flag
        else:
            raise NotImplementedError('self.mode is wrong: {}'.format(self.mode))

    def __getitem__(self, index):
        if 'semantic' not in self.mode:
            img_in, img_bg, img_rf, file_name, flag = self.get_data(index)
        else:
            img_in, img_bg, img_rf, seg_bg, seg_rf, file_name, flag = self.get_data(index)

        targets = {}

        img_in = torch.from_numpy(np.transpose(img_in, (2, 0, 1))).contiguous().float()
        targets['background'] = torch.from_numpy(np.transpose(img_bg, (2, 0, 1))).contiguous().float()
        if img_rf is not None:
            targets['reflection'] = torch.from_numpy(np.transpose(img_rf, (2, 0, 1))).contiguous().float()
        else:
            targets['reflection'] = None
        if 'semantic' in self.mode:
            seg_bg[seg_bg >= 21] = 0
            seg_rf[seg_rf >= 21] = 0

            targets['bg-semantic'] = torch.from_numpy(seg_bg).contiguous().long().unsqueeze(0)
            targets['rf-semantic'] = torch.from_numpy(seg_rf).contiguous().long().unsqueeze(0)

        targets['name'] = file_name

        img_in = (img_in - 0.5) * 2.
        targets['background'] = (targets['background'] - 0.5) * 2.
        targets['reflection'] = (targets['reflection'] - 0.5) * 2. if targets['reflection'] is not None else None

        return img_in, targets, flag


class VOCDataset(Dataset):
    def __init__(self, root_dir='', dataset_name='VOC2012', period='trainval', aug=False):
        self.dataset_name = root_dir
        self.root_dir = os.path.join(root_dir, 'VOCdevkit')
        self.dataset_dir = os.path.join(self.root_dir, dataset_name)
        self.rst_dir = os.path.join(self.root_dir, 'results', dataset_name, 'Segmentation')
        self.eval_dir = os.path.join(self.root_dir, 'eval_result', dataset_name, 'Segmentation')
        self.period = period
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.dataset_dir, 'Annotations')
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
        self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation')
        self.num_classes = 21
        if aug:
            file_name = self.set_dir + '/' + period + 'aug.txt'
        else:
            file_name = self.set_dir + '/' + period + '.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values

        if dataset_name == 'VOC2012':
            self.categories = [
                'aeroplane',        # 1
                'bicycle',          # 2
                'bird',             # 3
                'boat',             # 4
                'bottle',           # 5
                'bus',              # 6
                'car',              # 7
                'cat',              # 8
                'chair',            # 9
                'cow',              # 10
                'diningtable',      # 11
                'dog',              # 12
                'horse',            # 13
                'motorbike',        # 14
                'person',           # 15
                'pottedplant',      # 16
                'sheep',            # 17
                'sofa',             # 18
                'train',            # 19
                'tvmonitor']        # 20

            self.num_categories = len(self.categories)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name + '.jpg'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, c, _ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        if 'train' in self.period:
            seg_file = self.seg_dir + '/' + name + '.png'
            segmentation = np.array(Image.open(seg_file))
            sample['segmentation'] = segmentation

        return sample

    def onehot(self, label, num=None):
        num = self.num_classes if num is None else num
        m = label
        one_hot = np.eye(num)[m]
        return one_hot

