import pdb
import os
import time
import numpy as np
import pandas as pd
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint,model_device
from .metrics import MultiLabelReport
from pybert.config.basic_config import configs as config
from tqdm import tqdm

class Trainer(object):
    def __init__(self,model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 criterion,
                 evaluate_auc,
                 evaluate_f1,
                 incorrect,
                 lr_scheduler,
                 label_to_id,
                 verbose=1,
                 n_gpu            = None,
                 resume           = None,
                 model_checkpoint = None,
                 training_monitor = None,
                 TSBoard          = None,
                 early_stopping   = None,
                 gradient_accumulation_steps=1):
        self.model            = model              
        self.train_data       = train_data         
        self.val_data         = val_data           
        self.epochs           = epochs             
        self.optimizer        = optimizer          
        self.logger           = logger             
        self.verbose          = verbose            
        self.training_monitor = training_monitor   
        self.TSBoard          = TSBoard
        self.early_stopping   = early_stopping     # early_stopping
        self.resume           = resume             # resume best model or not
        self.model_checkpoint = model_checkpoint   # save mode
        self.evaluate_auc     = evaluate_auc       # evaluation metrics
        self.evaluate_f1      = evaluate_f1
        self.incorrect        = incorrect
        self.criterion        = criterion
        self.lr_scheduler     = lr_scheduler
        self.label_to_id      = label_to_id
        self.n_gpu            = n_gpu              
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._reset()

    def _reset(self):
        id_to_label = {value:key for key,value in self.label_to_id.items()}
        self.batch_num  = len(self.train_data)
        self.train_report = MultiLabelReport(id_to_label,sigmoid=True)
        self.valid_report = MultiLabelReport(id_to_label, sigmoid=True)
        self.progressbar = ProgressBar(n_batch = self.batch_num,loss_name='loss')
        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model,logger = self.logger)
        self.start_epoch = 1
        self.global_step = 0

        # resume best model 
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path = resume_path,model = self.model,optimizer = self.optimizer)
            self.model     = resume_list[0]
            self.optimizer = resume_list[1]
            best           = resume_list[2]
            self.start_epoch = resume_list[3]

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('trainable parameters: {:4}M'.format(params / 1000 / 1000))
        self.logger.info(self.model)

    # save mode
    def _save_info(self,epoch,val_loss):
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss,4)
        }
        return state

    def _valid_epoch(self, epoch):
        val_loss,count = 0, 0
        predicts   = []
        targets    = []
        self.model.eval()
        self.valid_report._reset()
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(self.val_data):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                target = label_ids.to(self.device)

                logits = self.model(input_ids, segment_ids,input_mask)
                loss = self.criterion(target=target, output=logits)
                val_loss += loss.item()
                predicts.append(logits)
                targets.append(target)
                count += 1
                self.valid_report.update(output=logits, target=target)

                self.TSBoard.set_step((epoch - 1) * len(self.val_data) + step, 'Valid')
                self.TSBoard.add_scalar('loss', loss.item())

            predicts = torch.cat(predicts,dim = 0)
            targets = torch.cat(targets,dim = 0)
            val_auc = self.evaluate_auc(output=predicts, target=targets)
            val_acc, val_PPV_macro, val_sen_macro, val_f1_macro,  val_PPV_micro, val_sen_micro, val_f1_micro = self.evaluate_f1(output=predicts, target=targets)
            if config['print_incorrect']:
                id2label = {v:k for k,v in self.label_to_id.items()}
                self.incorrect(output=predicts, target=targets, id2label=id2label, epochs=epoch)

        return {
            'val_loss': val_loss / count,
            'auc': val_auc,
            'f1_macro': val_f1_macro,
            'f1_micro': val_f1_micro,
            'PPV_macro': val_PPV_macro,
            'PPV_micro': val_PPV_micro,
            'sen_macro': val_sen_macro,
            'sen_micro': val_sen_micro,
            'acc': val_acc,
        }

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        self.train_report._reset()
        predicts, targets  = [], []
        for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(self.train_data):
            start = time.time()
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            target = label_ids.to(self.device)

            logits = self.model(input_ids, segment_ids,input_mask)
            loss = self.criterion(output=logits,target=target)
            self.train_report.update(output = logits,target = target)
            predicts.append(logits)
            targets.append(target)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward()
            # update learning
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lr_scheduler.step(training_step = self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            self.TSBoard.set_step((epoch - 1) * len(self.train_data) + step, 'Train')
            self.TSBoard.add_scalar('loss', loss.item())

            train_loss.update(loss.item(),input_ids.size(0))
            if self.verbose >= 1:
                self.progressbar.step(batch_idx= step,
                                      loss     = loss.item(),
                                      use_time = time.time() - start)

        predicts = torch.cat(predicts,dim = 0)
        targets = torch.cat(targets,dim = 0)
        auc = self.evaluate_auc(output=predicts, target=targets)
        acc, PPV_macro, sen_macro, f1_macro,  PPV_micro, sen_micro, f1_micro = self.evaluate_f1(output=predicts, target=targets)
        print("\ntraining result:")
        train_log = {
            'loss': train_loss.avg,
            'auc': auc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'PPV_macro': PPV_macro,
            'PPV_micro': PPV_micro,
            'sen_macro': sen_macro,
            'sen_micro': sen_micro,
            'acc': acc,
        }
        return train_log

    def _predict(self):
        predicts   = []
        self.model.eval()
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) \
            in tqdm(enumerate(self.val_data), total=len(self.val_data)):
                start = time.time()
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                target = label_ids.to(self.device)

                logits = self.model(input_ids, segment_ids,input_mask)
                predicts.append(logits)

            predicts = torch.cat(predicts,dim = 0)
            if config['multiclass']:
                predicts = [[1 if i > 0.7 else 2 if i > 0.5 else 3 if i > 0.3 else 0 for i in j] for j in predicts]
                predicts = np.array(predicts)
            else:
                predicts = np.round(predicts.sigmoid().data.cpu().numpy())

        return predicts
        

    def save_metrics(self, metrics, mode, epoch):
        for i in range(len(metrics[0])):
            self.TSBoard.set_step(epoch, mode + '_' + metrics[0][i])
            for label, result in metrics[1][i]:
                self.TSBoard.add_scalar(f'{label}', result)

    def train(self):
        if self.resume:
            print("\n----------------- start -----------------------")
            pred = self._predict()
            id_to_label = {v:k for k,v in self.label_to_id.items()}
            pred = pd.DataFrame(pred, columns=list(id_to_label.values()))
            data = pd.read_csv(config['raw_data_path'])
            sent = pd.concat([data,pred],axis=1)
            labvalue = ['', '', ' (Uncertain)', ' (Stable)']
            print("\n[Output] sentence.csv")
            output = []
            for row in tqdm(sent.values):
                label = pd.Series(row[2:])
                label_idx = label.nonzero()[0]
                label_name = []
                if len(label_idx)>0:
                    for n in label_idx:
                        v = int(label[n])
                        label_name.append(id_to_label[n]+labvalue[v])
                output.append((row[0], row[1], label_name))
            output = pd.DataFrame(output, columns=['ID', 'Sentence', 'Labels'])
            output.to_csv(os.path.join(config['result_dir'],'sentence.csv'), index = False)

            print("\n[Output] report.csv")
            report = sent.drop(['sentence'], axis=1)
            pre_idx, label, output = [], [], []
            for row in tqdm(report.values):
                idx = row[0].split('_')[0]
                if pre_idx == []:
                    label.append(row[1:])
                    pre_idx = idx
                elif idx == pre_idx:
                    label.append(row[1:])
                else: 
                    label = pd.DataFrame(label, columns=list(id_to_label.keys()))
                    label_idx = np.sum(label).nonzero()[0]
                    label_name = []
                    if len(label_idx)>0:
                        for n in label_idx:
                            v = int(max(label.iloc[:,n]))
                            label_name.append(id_to_label[n]+labvalue[v])
                    output.append((pre_idx, label_name))
                    pre_idx, label = idx, []
                    label.append(row[1:])
            output = pd.DataFrame(output, columns=['ID', 'Labels'])
            output.to_csv(os.path.join(config['result_dir'],'report.csv'), index=False)
            
        else:
            for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
                print("\n----------------- training start -----------------------")
                print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))
                train_log = self._train_epoch(epoch)
                if config['valid_size']:
                    val_log = self._valid_epoch(epoch)
                    logs = dict(train_log,**val_log)
                else: 
                    logs = train_log
                    logs['val_loss'] = logs['loss']
                self.logger.info('\nEpoch: %d - loss:%.4f, auc:%.4f, f1_macro:%.4f, f1_micro:%4f, acc:%.4f'%(
                            epoch,logs['loss'], logs['auc'],logs['f1_macro'],logs['f1_micro'], logs['acc']))

                print("---- train report every label -----")
                metrics = self.train_report.result()
                self.save_metrics(metrics, 'Train', epoch)
                if config['valid_size']:
                    print("---- valid report every label -----")
                    metrics = self.valid_report.result()
                    self.save_metrics(metrics, 'Valid', epoch)
                    self.TSBoard.set_step(epoch,'Valid')
                else:
                    self.TSBoard.set_step(epoch,'Train')
                self.TSBoard.add_scalar('AUC', logs['auc'])
                self.TSBoard.add_scalar('F1_macro', logs['f1_macro'])
                self.TSBoard.add_scalar('F1_micro', logs['f1_micro'])
                self.TSBoard.add_scalar('PPV_macro', logs['PPV_macro'])
                self.TSBoard.add_scalar('PPV_micro', logs['PPV_micro'])
                self.TSBoard.add_scalar('sen_macro', logs['sen_macro'])
                self.TSBoard.add_scalar('sen_micro', logs['sen_micro'])
                self.TSBoard.add_scalar('Accuracy', logs['acc'])

                if self.training_monitor:
                    self.training_monitor.step(logs)

                if self.model_checkpoint:
                    state = self._save_info(epoch,val_loss = logs['val_loss'])
                    self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)

                if self.early_stopping:
                    self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                    if self.early_stopping.stop_training:
                        break
