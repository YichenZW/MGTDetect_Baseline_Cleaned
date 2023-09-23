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


class GroverConstructor(object):
    def __init__(self, args):
        self.args = args

    def to_preprocessed(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError(
                "Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(
            self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(
            self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
    Raw data are formatted as:
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "table_id": datasets.Value("string"),
        "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                  "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
        "answer_text": datasets.features.Sequence(datasets.Value("string")),
    }
    """


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.balance_dataset = args.dataset.balance_dataset

        train_portion = args.dataset.train_portion
        if args.dataset.paired:
            cache_path = os.path.join(cache_root,
                                      'grover_' + args.model.task
                                      + "_{}".format(str(train_portion))
                                      + '_paired_train.cache')
            if os.path.exists(cache_path) and args.dataset.use_cache:
                self.extended_data = torch.load(cache_path)
            else:
                # TODO: Other reusable operations could be add here in the future about the graph
                self.extended_data = []
                data_dict = dict()
                for data in self.raw_datasets:
                    data_title = data["title"].lower()
                    if data_title in data_dict.keys():
                        data_dict[data_title].append(data)
                    else:
                        data_dict[data_title] = [data]
                pair_title = set()
                for i in data_dict:
                    if len(data_dict[i]) == 2:
                        pair_title.add(i)

                train_data_size = len(pair_title) * train_portion

                for idx, raw_data_title in tqdm(enumerate(data_dict)):
                    if raw_data_title in pair_title:
                        random_p = random.uniform(0, 1)
                        if random_p < 0.5:
                            continue
                        if len(self.extended_data) < train_data_size * 2:
                            pair_data = data_dict[raw_data_title]
                            self.extended_data.append(
                                raw_pair_data for raw_pair_data in pair_data)

                del data_dict

                if args.dataset.use_cache:
                    torch.save(self.extended_data, cache_path)

        else:
            cache_path = os.path.join(cache_root,
                                      'grover_' + args.model.task
                                      + "_{}".format(str(train_portion))
                                      + "_{}".format("balanced" if self.balance_dataset else "random")
                                      + "_human{}".format(str(args.dataset.human_portion) if self.balance_dataset else str(0))
                                      + '_train.cache')
            if os.path.exists(cache_path) and args.dataset.use_cache:
                self.extended_data = torch.load(cache_path)
                print("load data from cache {}".format(cache_path))
            else:
                # TODO: Other reusable operations could be add here in the future about the graph
                self.extended_data = []
                train_data_size = len(self.raw_datasets) * train_portion

                if self.balance_dataset:
                    human_num = train_data_size * args.dataset.human_portion
                    machine_num = train_data_size - human_num
                    category_data_size = [human_num, machine_num]
                    result_list = [0, 0]
                # for idx, raw_data in tqdm(enumerate(self.raw_datasets)):
                #     # increase randomness
                #     # random_p = random.uniform(0, 1)
                #     # if random_p < 0.5:
                #     #     continue
                #     label = 0 if raw_data["label"] == "human" else 1
                #     if self.balance_dataset:
                #         if result_list[label] < category_data_size[label]:
                #             result_list[label] += 1
                #             self.extended_data.append(raw_data)
                #     else:
                #         self.extended_data.append(raw_data)

                #     if len(self.extended_data) >= train_data_size:
                #         break
                idx_list = list(range(int(train_data_size)))
                random.shuffle(idx_list)
                for _, idx in tqdm(enumerate(idx_list)):
                    raw_data = self.raw_datasets[idx]
                    label = 0 if raw_data["label"] == "human" else 1
                    if self.balance_dataset:
                        if result_list[label] < category_data_size[label]:
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
        for i in range(len(train_data)):
            if train_data[i]['label'] == 'human':
                train_data[i]['title'] = 0
            else:
                train_data[i]['title'] = 1
        train_idx_by_label = {}
        for i in range(2):
            train_idx_by_label[i] = [idx for idx in range(
                len(train_data)) if train_data[idx]['title'] == i]

        return train_idx_by_label


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(
            cache_root, 'grover_' + args.model.task + '_dev.cache')
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


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(
            cache_root, 'grover_' + args.model.task + '_test.cache')
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
