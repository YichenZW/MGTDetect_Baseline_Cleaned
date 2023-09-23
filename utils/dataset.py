import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation framework.
    def __init__(self, args, training_args, tokenizer, raw_dataset, flags):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.raw_dataset = raw_dataset
        self.flags = flags

        # self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.raw_dataset[index]

        if "grover" in self.args.dataset.loader_path:
            seq_in = raw_item["article"]
            tokenized_input = self.tokenizer(
                seq_in,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
                # We found that set it as large as possible can boost the performance significantly
                # , meanwhile, due to the t5 uses a relative position coding, we need to manually
                # assign the max input length into some large numbers, instead of using the "max_model_length"
                # ,which the default is 512, which will hurt the performance a lot.
            )
            label = 0 if raw_item["label"] == "human" else 1
        elif "mnli" in self.args.dataset.loader_path:
            premise = raw_item["premise"]
            hypothesis = raw_item["hypothesis"]
            label = int(raw_item["label"])
            tokenized_input = self.tokenizer.encode_plus(premise,
                                                         hypothesis,
                                                         padding="max_length",
                                                         truncation=True,
                                                         max_length=self.training_args.input_max_length,
                                                         return_token_type_ids=True, )
        else: #gpt2
            # raise ValueError("Task Type not supported! (grover, mnli)")
            seq_in = raw_item["article"]
            tokenized_input = self.tokenizer(
                seq_in,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
                # We found that set it as large as possible can boost the performance significantly
                # , meanwhile, due to the t5 uses a relative position coding, we need to manually
                # assign the max input length into some large numbers, instead of using the "max_model_length"
                # ,which the default is 512, which will hurt the performance a lot.
            )
            label = 0 if raw_item["label"] == "human" else 1
            
        item = {
            'input_ids': torch.LongTensor(tokenized_input.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_input.data["attention_mask"]),
            'labels': torch.LongTensor([label]),
            "batch_idx": index,
            "flag": self.flags
        }
        return item

    def __len__(self):
        return len(self.raw_dataset)