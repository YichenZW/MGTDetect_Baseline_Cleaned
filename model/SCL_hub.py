import torch
import torch.nn.functional as F
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

        
class SCl(PreTrainedModel):
    def __init__(self, args, training_args, train_idx_by_label, dataset, task2num):

        self.args = args
        self.training_args = training_args
        num_label = task2num[args.dataset.loader_path]
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
        cls_feat = outputs_q.hidden_states[-1][:, 0, :]  # get [CLS] embedding

        q_value = outputs_q.logits
        ce_q_loss = outputs_q.loss

        # compute label-wise contrastive loss
        batch_idx_by_label = {}

        for i in range(2):
            batch_idx_by_label[i] = [idx for idx in range(batch_size) if int(labels[idx]) == i]  # for loss calculation

        # TODO: define train_idx_by_label
        contraloss = self.contrastive_loss_labelwise_winslide(batch_size, batch_idx_by_label, cls_feat)

        loss = (1.0 - self.args.model.contraloss_weight) * ce_q_loss + self.args.model.contraloss_weight * contraloss

        return {'logits': q_value, 'loss': loss}            

    def get_key(self, dic, value):
        return [k for k, v in dic.items() if value in v]            
                        
    def contrastive_loss_labelwise_winslide(self, batch_size, batch_idx_by_label, hidden_feats):
        '''
        hidden feats must be normalized
        '''
        # assert (len(batch_idx_by_label) == len(self.train_idx_by_label))
        hidden_feats = F.normalize(hidden_feats, dim=1)

        sim_matrix = torch.mm(hidden_feats, hidden_feats.T)         #(batch_size, batch_size)
        loss = 0.0

        for i in range(batch_size):

            label_list = self.get_key(batch_idx_by_label, i)
            label = label_list[0]
            one_same_label = torch.zeros((batch_size, )).cuda().scatter_(0, torch.tensor(batch_idx_by_label[label]).cuda(), 1.0)
            one_diff_label = torch.ones((batch_size, )).cuda().scatter_(0, torch.tensor(batch_idx_by_label[label]).cuda(), 0.0)
            one_for_not_i = torch.ones((batch_size,)).cuda().scatter_(0, torch.tensor([i]).cuda(), 0.0)  #[1, 1, 1, 0, 1, 1, 1] if i==3
            one_for_numerator = one_same_label.mul(one_for_not_i)
            
            numerator = torch.sum(one_for_numerator * torch.exp(sim_matrix[i, :] / self.args.model.temperature))
            denominator = torch.sum(one_for_not_i * torch.exp(sim_matrix[i, :] / self.args.model.temperature))

            if (numerator == 0): numerator += 1e-6

            loss += -torch.log(numerator / denominator)

        return loss / batch_size