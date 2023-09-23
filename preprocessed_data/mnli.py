import copy
import os
from copy import deepcopy

import torch
import random

from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm


class MnliConstructor(object):
    def __init__(self, args):
        self.args = args

    def to_preprocessed(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev matched, Dev mismatched sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_matched_dataset = DevDataset(self.args, raw_datasets['validation_matched'], cache_root)
        dev_mismatched_dataset = DevMisDataset(self.args, raw_datasets['validation_mismatched'], cache_root)

        return train_dataset, dev_matched_dataset, dev_mismatched_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.balance_dataset = args.dataset.balance_dataset

        train_portion = args.dataset.train_portion
        cache_path = os.path.join(cache_root,
                                  'mnli_' + args.model.task + "_{}".format(str(train_portion)) + '_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            # TODO: Other reusable operations could be add here in the future about the graph
            self.extended_data = []
            train_data_size = len(self.raw_datasets) * train_portion
            if self.balance_dataset:
                category_data_size = train_data_size // 3
                result_list = [0, 0, 0]
            # idx_list = list(range(train_data_size))
            # idx_list = random.sample(idx_list, int(train_data_size * train_portion))
            for idx, raw_data in tqdm(enumerate(self.raw_datasets)):
                # increase randomness
                random_p = random.uniform(0, 1)
                if random_p < 0.5:
                    continue
                label = raw_data["label"]
                if self.balance_dataset:
                    if result_list[label] < category_data_size:
                        result_list[label] += 1
                        self.extended_data.append(raw_data)
                else:
                    self.extended_data.append(raw_data)

                if len(self.extended_data) >= train_data_size:
                    break

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)

    def get_train_idx_by_label(self, train_data):
        train_idx_by_label = {}
        for i in range(3):
            train_idx_by_label[i] = [idx for idx in range(len(train_data)) if train_data[idx]['label'] == i]
        return train_idx_by_label


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'mnli_' + args.model.task + '_dev_match.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            # TODO: Other reusable operations could be add here in the future about the graph
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                self.extended_data.append(raw_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevMisDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'mnli_' + args.model.task + '_dev_mismatch.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            # TODO: Other reusable operations could be add here in the future about the graph
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                self.extended_data.append(raw_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
