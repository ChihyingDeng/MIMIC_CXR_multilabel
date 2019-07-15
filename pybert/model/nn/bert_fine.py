#encoding:utf-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pybert.config.basic_config import configs as config


class BertFinalPooler(nn.Module):
    def __init__(self, hidden_size, n=1):
        super(BertFinalPooler, self).__init__()
        self.dense = nn.Linear(hidden_size*n, hidden_size*n)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertFine(BertPreTrainedModel):
    def __init__(self,bertConfig,num_classes):
        super(BertFine ,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)
        n = 1
        if config['feature-based'] == 'Finetune_All': n = bertConfig.num_hidden_layers
        elif config['feature-based'] == 'Second_to_Last': n = bertConfig.num_hidden_layers-1
        elif config['feature-based'] == 'Concat_Last_Four': n = 4
        self.pooler = BertFinalPooler(bertConfig.hidden_size, n)
        self.classifier = nn.Linear(in_features=bertConfig.hidden_size*n, out_features=num_classes)
        self.apply(self.init_bert_weights)
        self.unfreeze_bert_encoder() 

    def freeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None, output_all_encoded_layers=True):
        encoded_layers, pooled_output = self.bert(input_ids,
                                        token_type_ids,
                                        attention_mask,
                                        output_all_encoded_layers=output_all_encoded_layers)
        
        if config['feature-based'] != 'Last':
            if config['feature-based'] == 'Finetune_All':
                sequence_output = torch.cat(encoded_layers,2)
            elif config['feature-based'] == 'First':
                sequence_output = encoded_layers[0]
            elif config['feature-based'] == 'Second_to_Last':
                sequence_output = torch.cat(encoded_layers[1:],1)
            elif config['feature-based'] == 'Sum_Last_Four':
                sequence_output = sum(encoded_layers[-4:])
            elif config['feature-based'] == 'Concat_Last_Four':
                sequence_output = torch.cat(encoded_layers[-4:],2)
            elif config['feature-based'] == 'Sum_All':
                sequence_output = sum(encoded_layers)
            pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
