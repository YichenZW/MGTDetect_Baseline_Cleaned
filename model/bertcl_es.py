import torch
import torch.nn.functional as F
import copy

from utils.dataset import TokenizedDataset
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, PreTrainedModel


# from .base import PushToHubFriendlyModel


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
    assert (hidden_batch_feats.size()[1] == queue.size()[1])

    queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)
    return queue


class BertCl(PreTrainedModel):
    def __init__(self, args, training_args, train_idx_by_label, dataset):

        self.args = args
        self.training_args = training_args
        config = AutoConfig.from_pretrained(args.model.name)
        super().__init__(config)
        self.train_idx_by_label = train_idx_by_label
        self.dataset = dataset

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.name, use_fast=False)
        self.model_q = AutoModelForSequenceClassification.from_pretrained(
            args.model.name,
            output_hidden_states=True,
        )
        self.model_k = copy.deepcopy(AutoModelForSequenceClassification.from_pretrained(
            args.model.name,
            output_hidden_states=True,
        ))
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        tokenized_dataset = TokenizedDataset(self.args, self.training_args, self.tokenizer, self.dataset, 0)
        train_loader = DataLoader(tokenized_dataset,
                                  batch_size=self.training_args.per_device_train_batch_size,
                                  )
        self.model_k.cuda()

        print('***start build queue')
        with torch.no_grad():
            for k, item in enumerate(train_loader):
                input_ids = item['input_ids'].cuda()
                attention_mask = item['attention_mask'].cuda()
                labels = item['labels'].cuda()
                output = self.model_k(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                init_feat = F.normalize(output.hidden_states[-1][:, 0, :], dim=1)
                if k == 0:
                    self.queue = init_feat
                else:
                    self.queue = torch.vstack((self.queue, init_feat))

        print(self.queue.size())
        print('***queue already builded')

        self.config = self.model_q.config
        self.feat_dim = self.config.hidden_size

    #         if args.special_tokens:
    #             self.tokenizer.add_tokens([v for k, v in args.special_tokens])
    #             self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs):

        selected_batch_idx = kwargs.pop("batch_idx", None)
        flag = kwargs.pop("flag", None)

        #         print('flag: ',flag)
        #         print('self_queue: ', self.queue)
        #         print('self_train_id: ', self.train_idx_by_label)

        if flag[0] == 1:
            batch_size = int(input_ids.size(0))

            outputs_q = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # TODO: Check if get last layer of [CLS] embedding correctly
            q_feat = outputs_q.hidden_states[-1][:, 0, :]  # get [CLS] embedding

            #         print('q_feat size: ', q_feat.size())
            q_value = outputs_q.logits
            ce_q_loss = outputs_q.loss

            # Update memory bank
            outputs_k = self.model_k(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            k_feat = outputs_k.hidden_states[-1][:, 0, :]
            # k_loss = outputs_k.loss

            self.dequeue_and_enqueue(k_feat, selected_batch_idx)

            # compute label-wise contrastive loss
            batch_idx_by_label = {}

            for i in range(2):
                batch_idx_by_label[i] = [idx for idx in range(batch_size) if labels[idx] == i]  # for loss calculation

            # TODO: define train_idx_by_label
            contraloss = self.contrastive_loss_es(batch_size, batch_idx_by_label, q_feat)

            # momentum update model k
            self.momentum_update(m=0.999)

            loss = (
                               1.0 - self.args.model.contraloss_weight) * ce_q_loss + self.args.model.contraloss_weight * contraloss

            return {'logits': q_value, 'loss': loss}

        if flag[0] == 2:
            eval_batch_size = int(input_ids.size(0))

            eval_outputs_q = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            eval_logits = eval_outputs_q.logits
            eval_q_feat = eval_outputs_q.hidden_states[-1][:, 0, :]
            eval_ce_q_loss = eval_outputs_q.loss

            eval_batch_idx_by_label = {}
            for i in range(2):
                eval_batch_idx_by_label[i] = [idx for idx in range(eval_batch_size) if
                                              labels[idx] == i]  # for loss calculation

            eval_contraloss = self.contrastive_loss_es(eval_batch_size, eval_batch_idx_by_label,
                                                                       eval_q_feat)
            eval_loss = (
                                    1.0 - self.args.model.contraloss_weight) * eval_ce_q_loss + self.args.model.contraloss_weight * eval_contraloss

            return {'logits': eval_logits, 'loss': eval_contraloss}

        if flag[0] == 3:
            test_batch_size = int(input_ids.size(0))

            test_outputs_q = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            test_logits = test_outputs_q.logits
            test_loss = test_outputs_q.loss

            return {'logits': test_logits, 'loss': test_loss}

    def contrastive_loss_es(self, batch_size, batch_idx_by_label, hidden_feats):
        hidden_feats = F.normalize(hidden_feats, dim=1)
        change_dic = {0:1, 1:0}
        loss = 0

        for i in batch_idx_by_label:
            q = hidden_feats[batch_idx_by_label[i]]
            pos_bank = self.queue[self.train_idx_by_label[i]]
            pos_pair = torch.mm(q, pos_bank.transpose(0,1))
            bottom_k = torch.topk(pos_pair, k=100, dim=1, largest=False).values
            neg_bank = self.queue[self.train_idx_by_label[change_dic[i]]]
            neg_pair = torch.mm(q, neg_bank.transpose(0,1))
            top_k = torch.topk(neg_pair, k=100, dim=1).values
            nominator = torch.sum(torch.exp(bottom_k / self.args.model.temperature), dim=1)
            denominator = torch.sum(torch.exp(top_k / self.args.model.temperature), dim=1) + nominator
            loss += torch.sum(-1.0 * torch.log(nominator / denominator))

        return loss / batch_size

    # def contrastive_loss_labelwise_winslide(self, batch_size, batch_idx_by_label, hidden_feats):
    #     '''
    #     hidden feats must bue normalized
    #
    #     '''
    #     assert (len(batch_idx_by_label) == len(self.train_idx_by_label))
    #     hidden_feats = F.normalize(hidden_feats, dim=1)
    #
    #     loss = 0
    #     for i in batch_idx_by_label:
    #         if (len(batch_idx_by_label) == 0):
    #             continue
    #         q = hidden_feats[batch_idx_by_label[i]]
    #         k = self.queue[self.train_idx_by_label[i]]
    #         l_pos = torch.sum(torch.exp(torch.mm(q, k.transpose(0, 1)) / self.args.model.temperature), dim=1)
    #         l_neg = torch.sum(torch.exp(torch.mm(q, self.queue.transpose(0, 1)) / self.args.model.temperature), dim=1)
    #         loss += torch.sum(-1.0 * torch.log(l_pos / l_neg))
    #     return loss / batch_size

    @torch.no_grad()
    def momentum_update(self, m=0.999):
        """
        encoder_k = m * encoder_k + (1 - m) encoder_q
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def dequeue_and_enqueue(self, hidden_batch_feats, selected_batch_idx):
        '''
        update memory bank by batch window slide; hidden_batch_feats must be normalized
        '''
        assert (hidden_batch_feats.size()[1] == self.queue.size()[1])

        self.queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)

