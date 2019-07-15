from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pdb
import csv
import numpy as np
from torch.utils.data import Dataset
from pybert.config.basic_config import configs as config


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid   = guid  
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids   = input_ids   # idex of tokens
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id

class CreateDataset(Dataset):
    def __init__(self,data_path,max_seq_len,tokenizer,example_type,seed):
        self.seed    = seed
        self.max_seq_len  = max_seq_len
        self.example_type = example_type
        self.data_path  = data_path
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.build_examples()

    def read_data(self,quotechar = None):
        '''
        default: tab (\t)
        :param quotechar:
        :return:
        '''
        lines = []
        with open(self.data_path,'r',encoding='utf-8') as fr:
            reader = csv.reader(fr,delimiter = '\t',quotechar = quotechar)
            for line in reader:
                lines.append(line)
        return lines

    def build_examples(self):
        lines = self.read_data()
        self.examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(self.example_type,i)
            if config['resume']:
                label = [0]
            else:
                label = [np.float32(x) for x in line[0].split(',')] 
            text_a = line[1]
            example = InputExample(guid = guid,text_a = text_a,label= label)
            self.examples.append(example)
        del lines

    def build_features(self,example):
        '''
        #  [Two sentences]:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        #  [One sentence]:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        '''
        features = []
        input_data, sent_ids, sent_id = [], [], True
        
        #tokenization
        tokens_a = self.tokenizer.tokenize(example.text_a)   
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:(self.max_seq_len - 2)]
        # add CLS token
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)  # type_ids
        # convert token to token's idex
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # mask
        input_mask = [1] * len(input_ids)
        # padding
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        # label
        label_id = example.label
        feature = InputFeature(input_ids = input_ids,input_mask = input_mask,
                               segment_ids = segment_ids,label_id = label_id)
        return feature

    def _preprocess(self,index):
        example = self.examples[index]
        feature = self.build_features(example)

        return np.array(feature.input_ids),np.array(feature.input_mask),\
               np.array(feature.segment_ids),np.array(feature.label_id)

    def __getitem__(self, index):
        return self._preprocess(index)

    def __len__(self):
        return len(self.examples)
