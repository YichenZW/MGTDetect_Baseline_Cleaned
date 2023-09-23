import torch

from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, PreTrainedModel
# from .base import PushToHubFriendlyModel


class BaseModel(PreTrainedModel):
    def __init__(self, args, task2num):

        self.args = args
        num_label = task2num[args.dataset.loader_path]
        config = AutoConfig.from_pretrained(args.model.name, num_labels=num_label)
        super().__init__(config)
        padding_token = config.vocab_size

        # Load tokenizer and model.
        if 'gpt2' in args.model.name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model.name, use_fast=False, num_labels=num_label, cache_dir = './gpt2_tokenizer',
                                        pad_token='[PAD]')
            self.pretrain_model = AutoModelForSequenceClassification.from_pretrained(
                args.model.name,
                num_labels=num_label,
                pad_token_id=padding_token,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('./roberta_tokenizer/checkpoint-567', use_fast=False, num_labels=num_label)
            self.pretrain_model = AutoModelForSequenceClassification.from_pretrained(
                args.model.name,
                num_labels=num_label,
            )

            # for param in self.pretrain_model.roberta.parameters():
            #     param.requires_grad = False

        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
    #
    # def generate(self, input_ids, attention_mask, **kwargs):
    #     generated_ids = self.pretrain_model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         use_cache=True,
    #         **kwargs,
    #     )
    #
    #     return generated_ids