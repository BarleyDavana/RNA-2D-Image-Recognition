import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
from PIL import Image

import shutil

import cv2
from augment import Augment

def get_train_file_path(image_id, urs_id):
    
    is_file_moved = True
    
    dir_index = image_id // 500
    dir_path = "/data1/casual/luyitan/a325/data5w/train8w/{}".format(dir_index)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    if not is_file_moved:
        src_file_path = "/data1/casual/luyitan/a325/data5w/train8w/{}.png".format(urs_id)

        dest_file_path = "{}/{}.png".format(dir_path, urs_id)
        shutil.move(src_file_path, dest_file_path)
        is_file_moved = True
    return "/data1/casual/luyitan/a325/data5w/train8w/{}/{}.png".format(
        dir_index, urs_id
    )

def get_test_file_path(urs_id):
    return "/data1/casual/luyitan/a325/data1/testdb/{}.png".format(
    # return "/data1/casual/luyitan/a325/data5w/test/{}.png".format(
        urs_id
    )

def get_val_file_path(urs_id):
    return "/data1/casual/luyitan/a325/data5w/val/{}.png".format(
        urs_id
    )

def pil_loader(path: str) -> Image.Image:
    if not os.path.exists(path) or not os.path.getsize(path):
        return None
    with open(path, 'rb') as f:
        img = Image.open(f)
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        else:
            return img.convert('RGB')

class GetDataset(Dataset):
    def __init__(self, transform, caption_file, type='train', vocab=None):
        self.loader = pil_loader
        self.transform = transform
        self.caption_file = pd.read_csv(caption_file)
        if type == 'train':
            self.caption_file['file_path'] = self.caption_file.progress_apply(lambda row: get_train_file_path(row['image_id'], row['urs_id']), axis=1)
        elif type == 'test':
            self.caption_file['file_path'] = self.caption_file['urs_id'].progress_apply(get_test_file_path)
        elif type == 'val':
            self.caption_file['file_path'] = self.caption_file['urs_id'].progress_apply(get_val_file_path)
        self.vocab = vocab
        self.augmentor = Augment()

    def __len__(self):
        return len(self.caption_file)

    def __getitem__(self, idx):
        sample = self.loader(self.caption_file['file_path'][idx])
        if sample:
            #sample = self.augmentor.apply(sample)
            sample = self.transform(sample)
        caption_vec = []
        caption_vec += self.vocab.string_to_ints1(self.caption_file['structure'][idx])
        #caption_vec += self.vocab.string_to_ints2(self.caption_file['merged'][idx])
        target_vec = []
        target_vec += self.vocab.string_to_ints(self.caption_file['structure'][idx])

        return sample, torch.tensor(caption_vec), torch.tensor(target_vec)

class CapsCollate:
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = []
        targets = []
        captions = []

        for item in batch:
            imgs.append(item[0].unsqueeze(0))
            targets.append(item[2])
            captions.append(item[1])
        imgs = torch.cat(imgs, dim=0)
        captions = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, captions, targets
