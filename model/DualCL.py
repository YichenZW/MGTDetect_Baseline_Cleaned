import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, PreTrainedModel
# from .base import PushToHubFriendlyModel


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
    assert(hidden_batch_feats.size()[1] == queue.size()[1])

    queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)
    return queue

        
class DualCl(PreTrainedModel):
    def __init__(self, args, training_args, train_idx_by_label, dataset, task2num):

        self.args = args
        self.training_args = training_args
        num_label = task2num[args.dataset.loader_path]
        self.num_label = num_label
        config = AutoConfig.from_pretrained(args.model.name, num_labels=num_label)
        super().__init__(config)
        self.train_idx_by_label = train_idx_by_label
        self.dataset = dataset

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.name, use_fast=False, config=config)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model.name,
            output_hidden_states=True,
            num_labels=num_label
        )

        self.config = config
        self.feat_dim = self.config.hidden_size

        self.temp = 0.1
        self.xent_loss = nn.CrossEntropyLoss()


#         if args.special_tokens:
#             self.tokenizer.add_tokens([v for k, v in args.special_tokens])
#             self.pretrain_model.resize_token_embeddings(len(self.tokenizer))           
    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs):

        selected_batch_idx = kwargs.pop("batch_idx", None)

        #         print('flag: ',flag)
        #         print('self_queue: ', self.queue)
        #         print('self_train_id: ', self.train_idx_by_label)

        batch_size = int(input_ids.size(0))

        outputs_q = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # TODO: Check if get last layer of [CLS] embedding correctly
        cls_feats = outputs_q.hidden_states[-1][:, 0, :]  # get [CLS] embedding
        hiddens = outputs_q.hidden_states[-1]
        label_feats = hiddens[:, 1: self.num_label+1, :]
        predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        labels = labels.reshape(-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=labels.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        ce_loss = 0.5 * self.xent_loss(outputs['predicts'], labels) / 10
        cl_loss_1 = 0.25 * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, labels)
        cl_loss_2 = 0.25 * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, labels)
        loss = ce_loss + cl_loss_1 + cl_loss_2        
        q_value = outputs_q.logits
        ce_q_loss = outputs_q.loss
        return {'logits': q_value, 'loss': loss}            

    def nt_xent_loss(self, anchor, target, labels):
            with torch.no_grad():
                labels = labels.unsqueeze(-1)
                mask = torch.eq(labels, labels.transpose(0, 1))
                # delete diag elem
                mask = mask ^ torch.diag_embed(torch.diag(mask))
            # compute logits
            anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
            # delete diag elem
            anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
            logits = anchor_dot_target - logits_max.detach()
            # compute log prob
            exp_logits = torch.exp(logits)
            # mask out positives
            logits = logits * mask
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
            # in case that mask.sum(1) is zero
            mask_sum = mask.sum(dim=1)
            mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
            # compute log-likelihood
            pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
            loss = -1 * pos_logits.mean()
            return loss
