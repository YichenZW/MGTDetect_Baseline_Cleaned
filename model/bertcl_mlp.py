import torch
import torch.nn.functional as F
import copy

from utils.dataset import TokenizedDataset
from torch import nn
from torch.utils.data import DataLoader,SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, PreTrainedModel, AutoModel
# from .base import PushToHubFriendlyModel


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
    assert(hidden_batch_feats.size()[1] == queue.size()[1])

    queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)
    return queue

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        
        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)

class Encoder(PreTrainedModel):
    #add num_mlp_layers, hidden_dim and out_dim to trainning_args
    def __init__(self, args, training_args):
        self.args = args
        self.training_args = training_args
        config = AutoConfig.from_pretrained(args.model.name)
        super().__init__(config)
        
        self.bert = AutoModel.from_pretrained(args.model.name)
        self.feat_dim = list(self.bert.modules())[-2].out_features
        self.mlp = MLP(self.training_args.num_layers, self.feat_dim, self.training_args.hidden_dims, self.training_args.out_dims)
        
    def forward(self, input_ids, attention_mask):
        encoding = self.mlp(self.bert(input_ids, attention_mask)[0][:, 0])
        cls_rep = self.bert(input_ids, attention_mask)[0][:, 0]
        return encoding, cls_rep


class BertCl(PreTrainedModel):
    def __init__(self, args, training_args, train_idx_by_label, dataset):

        self.args = args
        self.training_args = training_args
        config = AutoConfig.from_pretrained(args.model.name)
        super().__init__(config)
        self.train_idx_by_label = train_idx_by_label
        self.dataset = dataset
        self.classifier = torch.nn.Linear(768, 2)
        self.criterion = torch.nn.CrossEntropyLoss()


        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.name, use_fast=False)
        self.model_q = Encoder(self.args, self.training_args)
        self.model_k = copy.deepcopy(Encoder(self.args, self.training_args))
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        tokenized_dataset = TokenizedDataset(self.args, self.training_args, self.tokenizer, self.dataset, 0)
        train_loader = DataLoader(tokenized_dataset,
                batch_size=self.training_args.per_device_train_batch_size,
                )
        self.model_k.cuda()
        
        print('***start build queue***')
        with torch.no_grad():
            for k, item in enumerate(train_loader):
                input_ids = item['input_ids'].cuda()
                attention_mask = item['attention_mask'].cuda()
                labels = item['labels'].cuda()
                output, cls = self.model_k(input_ids = input_ids, attention_mask = attention_mask)
                init_feat = F.normalize(output, dim=1)
                if k==0:
                    self.queue = init_feat
                else:
                    self.queue = torch.vstack((self.queue, init_feat))
                    
        print(self.queue.size())
        print('***queue already builded***')
                           
        self.config = self.model_q.config
        self.feat_dim = self.config.hidden_size


#         if args.special_tokens:
#             self.tokenizer.add_tokens([v for k, v in args.special_tokens])
#             self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                batch_idx,
                flag,
                **kwargs):
        
        selected_batch_idx = batch_idx
        labels_cal = torch.squeeze(labels)

#         print('flag: ',flag)
#         print('self_queue: ', self.queue)
#         print('self_train_id: ', self.train_idx_by_label)

        if flag[0] == 1:
            batch_size = int(input_ids.size(0))

            outputs_q, cls_rep = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = self.classifier(cls_rep)
#             print(labels.size())
            ce_q_loss = self.criterion(logits, labels_cal)       

            # Update memory bank
            outputs_k, cls_rep_k = self.model_k(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # k_loss = outputs_k.loss

            self.dequeue_and_enqueue(outputs_k, selected_batch_idx)

            # compute label-wise contrastive loss
            batch_idx_by_label = {}

            for i in range(2):
                batch_idx_by_label[i] = [idx for idx in range(batch_size) if labels[idx] == i] #for loss calculation

            # TODO: define train_idx_by_label
            contraloss = self.contrastive_loss_labelwise_winslide(batch_size, batch_idx_by_label, outputs_q)

            # momentum update model k
            self.momentum_update(m=0.999)

            loss = (1.0 - self.args.model.contraloss_weight) * ce_q_loss + self.args.model.contraloss_weight * contraloss

            return {'logits': logits, 'loss': loss}
        
        if flag[0] == 2:
            eval_batch_size = int(input_ids.size(0))
            

            eval_outputs_q, cls_rep = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            eval_logits = self.classifier(cls_rep)
            eval_q_feat = eval_outputs_q
            eval_ce_q_loss = ce_q_loss = self.criterion(eval_logits, labels_cal)  
            
            
            eval_batch_idx_by_label = {}
            for i in range(2):
                eval_batch_idx_by_label[i] = [idx for idx in range(eval_batch_size) if labels[idx] == i] #for loss calculation
            
            eval_contraloss = self.contrastive_loss_labelwise_winslide(eval_batch_size, eval_batch_idx_by_label, eval_q_feat)
            eval_loss = (1.0 - self.args.model.contraloss_weight) * eval_ce_q_loss + self.args.model.contraloss_weight * eval_contraloss
           
            
            return {'logits': eval_logits, 'loss': eval_contraloss}
        
        if flag[0] == 3:
            test_batch_size = int(input_ids.size(0))
            
            test_outputs_q, cls_rep = self.model_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            test_logits = self.classifier(cls_rep)
            test_loss = ce_q_loss = self.criterion(test_logits, labels_cal)  
            
            
            return {'logits': test_logits, 'loss': test_loss}
            
            


    
    def contrastive_loss_labelwise_winslide(self, batch_size, batch_idx_by_label, hidden_feats):
        '''
        hidden feats must bue normalized
        
        '''
        assert(len(batch_idx_by_label) == len(self.train_idx_by_label))
        hidden_feats = F.normalize(hidden_feats, dim=1)
        
        loss = 0
        for i in batch_idx_by_label:
            if(len(batch_idx_by_label)==0):
                continue
            q = hidden_feats[batch_idx_by_label[i]]
            k = self.queue[self.train_idx_by_label[i]]
            l_pos = torch.sum(torch.exp(torch.mm(q, k.transpose(0,1))/self.args.model.temperature), dim=1)
            l_neg = torch.sum(torch.exp(torch.mm(q, self.queue.transpose(0,1))/self.args.model.temperature), dim=1)
            loss += torch.sum(-1.0*torch.log(l_pos/l_neg))
        return loss/batch_size

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
        