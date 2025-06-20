import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule  # ✅ Sửa từ pytorch_lightning -> lightning.pytorch

from tamer.datamodule.dataset import HMEDataset
from .vocab import vocab

Data = List[Tuple[str, np.ndarray, List[str]]]

def data_iterator(
    data: Data,
    batch_size: int,
    max_size: int,
    is_train: bool,
    maxlen: int = 200,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].shape[0] * x[1].shape[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.shape[0] * fea.shape[1]
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)

        if is_train and len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif is_train and size > max_size:
            print(f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {max_size}, ignore")
        else:
            if batch_image_size > max_size or i == batch_size:
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                fname_batch = [fname]
                feature_batch = [fea]
                label_batch = [lab]
                biggest_image_size = size
                i = 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    if fname_batch:
        fname_total.append(fname_batch)
        feature_total.append(feature_batch)
        label_total.append(label_batch)

    print("total", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(folder: str, dir_name: str) -> Data:
    with open(os.path.join(folder, dir_name, "images.pkl"), "rb") as f:
        images = pickle.load(f)
    with open(os.path.join(folder, dir_name, "caption.txt"), "r") as f:
        captions = f.readlines()

    data = []
    for line in captions:
        tmp = line.strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        img = images[img_name]
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")
    return data


@dataclass
class Batch:
    img_bases: List[str]
    imgs: FloatTensor
    mask: LongTensor
    indices: List[List[int]]

    def __len__(self):
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]
    n_samples = len(images_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

    for idx, s_x in enumerate(images_x):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 0

    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(folder: str, split: str, batch_size: int, max_size: int, is_train: bool):
    data = extract_data(folder, split)
    return data_iterator(data, batch_size, max_size, is_train)


class HMEDatamodule(LightningDataModule):  # ✅ Sửa kế thừa
    def __init__(
        self,
        folder: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data/crohme",
        test_folder: str = "2014",
        max_size: int = 32e4,
        scale_to_limit: bool = True,
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.test_folder = test_folder
        self.max_size = max_size
        self.scale_to_limit = scale_to_limit
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        vocab.init(os.path.join(folder, "dictionary.txt"))
        print(f"Load data from: {self.folder}")

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = HMEDataset(
                build_dataset(self.folder, "train", self.train_batch_size, self.max_size, True),
                True,
                self.scale_aug,
                self.scale_to_limit,
            )
            self.val_dataset = HMEDataset(
                build_dataset(self.folder, self.test_folder, self.eval_batch_size, self.max_size, False),
                False,
                self.scale_aug,
                self.scale_to_limit,
            )

        if stage in (None, "test"):
            self.test_dataset = HMEDataset(
                build_dataset(self.folder, self.test_folder, self.eval_batch_size, self.max_size, False),
                False,
                self.scale_aug,
                self.scale_to_limit,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # each element from data_iterator is already a full batch
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
