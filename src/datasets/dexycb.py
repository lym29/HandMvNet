import os
import math
from typing import List, Union

import braceexpand
import lightning as L
import webdataset as wds

from datasets.ho3d import HO3DSamplePreprocessor
from datasets.utils import webdataset_nodesplitter


class DexYCBMultiview:
    def __init__(self, config):
        self.name = type(self).__name__
        self.cfg = config
        self.data_urls = {
            "train": os.path.join(self.cfg["dataset_dir"], "DexYCB_mv_train-{000000..000019}.tar"),
            "val": os.path.join(self.cfg["dataset_dir"], "DexYCB_mv_val-{000000..000001}.tar"),
            "test": os.path.join(self.cfg["dataset_dir"], "DexYCB_mv_test-{000000..000003}.tar"),
        }

    def expand_urls(self, urls: Union[str, List[str]]):
        if isinstance(urls, str):
            urls = [urls]
        urls = [
            u
            for url in urls
            for u in braceexpand.braceexpand(os.path.expanduser(os.path.expandvars(url)))
        ]
        return urls

    def get_dataset(self, data_split="train"):
        assert data_split in ["train", "test", "val"], f"{self.name} unsupported data split {data_split}"

        urls = self.expand_urls(self.data_urls[data_split])
        dataset = wds.WebDataset(
            urls=urls,
            nodesplitter=webdataset_nodesplitter(len(urls)),
            workersplitter=wds.split_by_worker,
            shardshuffle=data_split == "train",
            resampled=data_split == "train",
        )

        if data_split == "train":
            print(f"[Dangerous] Resampled={data_split == 'train'}, Mode={data_split}")
            dataset = dataset.shuffle(500)

        dataset = dataset.decode("rgb8")
        processor = HO3DSamplePreprocessor(self.cfg, subset=data_split)
        dataset = dataset.map(processor)

        return dataset


class DexYCBDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.cfg.setdefault("total_views", 8)
        self.cfg.setdefault("root_idx", 2)
        self.cfg.setdefault("input_res", (480, 640))
        self.cfg.setdefault("scale", 1000)

        dexycb = DexYCBMultiview(self.cfg)
        self.train_samples = self.cfg.get("train_samples", 25387)
        self.val_samples = self.cfg.get("val_samples", 1412)
        self.test_samples = self.cfg.get("test_samples", 4951)
        self.train_set = dexycb.get_dataset(data_split="train")
        self.val_set = dexycb.get_dataset(data_split="val")
        self.test_set = dexycb.get_dataset(data_split="test")

    def train_dataloader(self):
        return wds.WebLoader(
            self.train_set,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        ).with_epoch(self.train_samples // self.cfg["batch_size"]).shuffle(self.cfg["batch_size"] * 2)

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_set,
            batch_size=self.cfg["batch_size"],
            num_workers=1,
            pin_memory=True,
        ).with_epoch(math.ceil(self.val_samples / self.cfg["batch_size"]))

    def test_dataloader(self):
        return wds.WebLoader(
            self.test_set,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        ).with_epoch(math.ceil(self.test_samples / self.cfg["batch_size"]))

    def predict_dataloader(self):
        return wds.WebLoader(
            self.test_set,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        ).with_epoch(math.ceil(self.test_samples / self.cfg["batch_size"]))
