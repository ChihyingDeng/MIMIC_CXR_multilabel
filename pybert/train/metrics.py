import os
import pandas as pd
import torch
import numpy as np
import csv
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_recall_fscore_support
import pdb
from pybert.config.basic_config import configs as config

class Accuracy(object):
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

class AUC(object):
    def __init__(self,sigmoid):
        self.sigmoid = sigmoid

    def __call__(self,output,target):
        target = target.cpu().numpy()
        if self.sigmoid:
            y_pred = output.sigmoid().data.cpu().numpy()
            if config['multiclass']:
                y_pred = [[1 if i > 0.7 else 2 if i > 0.5 else 3 if i > 0.3 else 0 for i in j] for j in y_pred]
                y_pred = np.array(y_pred)
            else:
                y_pred = np.round(y_pred)
        else:
            y_pred = output.data.cpu().numpy()
        try: auc = roc_auc_score(y_score=y_pred,y_true=target)
        except ValueError: auc = 0
        return np.mean(auc)

class F1Score(object):
    def __init__(self,sigmoid):
        self.sigmoid = sigmoid
    def __call__(self,output,target):
        if self.sigmoid:
            y_pred = output.sigmoid().data.cpu().numpy()
            if config['multiclass']:
                y_pred = [[1 if i > 0.7 else 2 if i > 0.5 else 3 if i > 0.3 else 0 for i in j] for j in y_pred]
                y_pred = np.array(y_pred)
            else:
                y_pred = np.round(y_pred)
        else:
            y_pred = output.data.cpu().numpy()
        y_true = target.cpu().numpy()
        try:
            PPV_macro, sen_macro, f1_macro, support_macro = \
            precision_recall_fscore_support(y_true, y_pred, average='micro')
            PPV_micro, sen_micro, f1_micro, support_micro = \
            precision_recall_fscore_support(y_true, y_pred, average=None)    
            PPV_micro, sen_micro, f1_micro = np.mean(PPV_micro), np.mean(sen_micro), np.mean(f1_micro)
            correct = np.sum((y_true == y_pred).astype(int))
            acc = correct / (y_pred.shape[0]*y_pred.shape[1])
        except ValueError: 
            acc, PPV_macro, sen_macro, f1_macro, PPV_micro, sen_micro, f1_micro =\
            0,0,0,0,0,0,0

        return acc, PPV_macro, sen_macro, f1_macro, PPV_micro, sen_micro, f1_micro 

class Incorrect(object):
    def __init__(self,sigmoid):
        self.sigmoid = sigmoid

    def __call__(self,output,target, id2label, epochs):
        target = target.cpu().numpy()
        if self.sigmoid:
            output = output.sigmoid().data.cpu().numpy()
            output = np.round(output)
        else:
            output = output.data.cpu().numpy()
        sent = []
        with open(config['valid_file_path'],'r') as f:
            reader = csv.reader(f,delimiter = '\t',quotechar = None)
            for line in reader:
                sent.append(line[1])

        if epochs%5 == 0:
            pathname = os.path.join(config['misclassified_dir'], config['feature-based'], str(config['seed']))
            if not os.path.isdir(pathname): os.makedirs(pathname)
            error = []
            for i in range(output.shape[0]):
                p = True
                for j in range(output.shape[1]):
                    if output[i][j] != target[i][j]: 
                        if p:
                            p = False
                            error.append([sent[i],id2label[j],target[i][j]])
                        else:
                            error.append(['',id2label[j],target[i][j]])
            error = pd.DataFrame(error, columns=['sentence','label','correct_value'])
            error.to_csv(pathname+'/epo_'+str(epochs)+'.csv', index=False)


class ClassReport(object):
    def __init__(self,target_names = None):
        self.target_names = target_names

    def __call__(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        classify_report = classification_report(y_true, y_pred,target_names=self.target_names)
        print('\n\nclassify_report:\n', classify_report)


class MultiLabelReport(object):
    def __init__(self,id_to_label,sigmoid):
        self.id_to_label = id_to_label
        self.sigmoid = sigmoid
        self._reset()

    def _reset(self):
        self.outputs = None
        self.targets = None

    def _compute_f1(self,output,target):
        PPV_macro, sen_macro, f1_macro, support_macro = \
            precision_recall_fscore_support(target, output, average="macro")
        PPV_micro, sen_micro, f1_micro, support_micro = \
            precision_recall_fscore_support(target, output, average="micro")
        return PPV_macro, sen_macro, f1_macro, PPV_micro, sen_micro, f1_micro

    def _compute_auc(self,output,target):
        try: auc =roc_auc_score(y_score=output,y_true=target)
        except ValueError: auc = 0
        return auc

    def _compute_acc(self,output,target):
        correct = np.sum((target == output).astype(int))
        acc = correct / output.shape[0]

        return acc

    def result(self):
        AUC_all, F1_micro_all, ACC_all = [], [], []
        PPV_micro_all, sen_micro_all, support_micro_all, labels = [], [], [], []

        if config['multiclass']:
            F1_macro_all, PPV_macro_all, sen_macro_all = [], [], []
            self.outputs = [[1 if i > 0.7 else 2 if i > 0.5 else 3 if i > 0.3 else 0 for i in j] for j in self.outputs]
            self.outputs = np.array(self.outputs)
        else:
            self.outputs = np.round(self.outputs)

        for i, label in self.id_to_label.items():
            auc = self._compute_auc(self.outputs[:, i], self.targets[:, i])
            AUC_all.append([label, auc])
            acc = self._compute_acc(self.outputs[:, i], self.targets[:, i])
            ACC_all.append([label, acc])
            labels.append(label)
            if config['multiclass']:
                PPV_macro, sen_macro, f1_macro, PPV_micro, sen_micro, f1_micro =\
                self._compute_f1(self.outputs[:, i], self.targets[:, i])
                F1_macro_all.append([label, f1_macro])
                F1_micro_all.append([label, f1_micro])
                PPV_macro_all.append([label, PPV_macro])
                PPV_micro_all.append([label, PPV_micro])
                sen_macro_all.append([label, sen_macro])
                sen_micro_all.append([label, sen_micro])
        if not config['multiclass']:   
            PPV_micro, sen_micro, f1_micro, support_micro = \
            precision_recall_fscore_support(self.targets, self.outputs, average=None)        
            F1_micro_all = list(zip(labels, list(f1_micro)))
            PPV_micro_all = list(zip(labels, list(PPV_micro)))
            sen_micro_all = list(zip(labels, list(sen_micro)))
            support_micro_all = list(zip(labels, list(support_micro)))

        if config['multiclass']:
            metrics = [['AUC', 'F1_macro', 'F1_micro',  'Accuracy', \
                        'PPV_macro', 'PPV_micro', 'sen_macro', 'sen_micro'], \
                       [AUC_all, F1_macro_all, F1_micro_all, ACC_all, \
                       PPV_macro_all, PPV_micro_all, sen_macro_all, sen_micro_all]]
        else:
             metrics = [['AUC', 'F1_micro',  'Accuracy', 'PPV_micro', 'sen_micro', 'support_micro'], \
                       [AUC_all, F1_micro_all, ACC_all, PPV_micro_all, sen_micro_all, support_micro_all]]

        return metrics

    def update(self,output,target):
        target = target.cpu().numpy()
        if self.sigmoid:
            logits = output.sigmoid().data.cpu().numpy()
        else:
            logits = output.data.cpu().numpy()
        if self.outputs is None:
            self.outputs = logits
            self.targets = target
        else:
            self.outputs = np.concatenate((self.outputs,logits),axis =0)
            self.targets = np.concatenate((self.targets, target), axis=0)
