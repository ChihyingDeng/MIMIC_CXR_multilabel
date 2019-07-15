#encoding:utf-8
import os
from os import path
import pandas as pd
import torch
import warnings
import datetime
from json import load
from pytorch_pretrained_bert.optimization import BertAdam
from pybert.train.metrics import AUC, F1Score, Incorrect
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.dataset import CreateDataset
from pybert.io.data_transformer import DataTransformer
from pybert.io.split_reports import SplitReports
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything
from pybert.config.basic_config import configs as config
from pybert.callback.lrscheduler import BertLr
from pybert.model.nn.bert_fine import BertFine
from pybert.preprocessing.preprocessor import Preprocessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.callback.writetensorboard import WriterTensorboardX
from pytorch_pretrained_bert.tokenization import BertTokenizer
warnings.filterwarnings("ignore")

def main():
    # **************************** Basic Info  ***********************
    logger = init_logger(log_name=config['arch'], log_dir=config['log_dir'])
    logger.info("seed is %d"%config['seed'])
    device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'
    seed_everything(seed=config['seed'],device=device)
    logger.info('starting load data from disk')

    # split the reports
    if config['resume']:
        split_reports = SplitReports(raw_reports_dir = config['raw_reports_dir'],
                                     raw_data_path   = config['raw_data_path'])
        split_reports.split()

    df = pd.read_csv(config['raw_data_path'])
    label_list = df.columns.values[2:].tolist()
    config['label_to_id'] = {label : i for i, label in enumerate(label_list)}
    config['id_to_label'] = {i : label for i, label in enumerate(label_list)}
    config['vocab_path'] = path.sep.join([config['bert_model_dir'], 'vocab.txt'])

    # **************************** Data  ***********************
    data_transformer = DataTransformer(logger      = logger,
                                       raw_data_path=config['raw_data_path'],
                                       label_to_id = config['label_to_id'],
                                       train_file  = config['train_file_path'],
                                       valid_file  = config['valid_file_path'],
                                       valid_size  = config['valid_size'],
                                       seed        = config['seed'],
                                       preprocess  = Preprocessor(),
                                       shuffle     = config['shuffle'],
                                       skip_header = True,
                                       stratify    = False)
    # dataloader and pre-processing
    data_transformer.read_data()

    tokenizer = BertTokenizer(vocab_file=config['vocab_path'],do_lower_case=config['do_lower_case'])

    # train
    train_dataset   = CreateDataset(data_path    = config['train_file_path'],
                                    tokenizer    = tokenizer,
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'train')
    # valid
    valid_dataset   = CreateDataset(data_path    = config['valid_file_path'],
                                    tokenizer    = tokenizer,
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'valid')
    # resume best model
    if config['resume']:
        train_loader =  [0]
    else:
        train_loader = DataLoader(dataset     = train_dataset,
                                  batch_size  = config['batch_size'],
                                  num_workers = config['num_workers'],
                                  shuffle     = True,
                                  drop_last   = False,
                                  pin_memory  = False)
    # valid
    valid_loader = DataLoader(dataset     = valid_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = False,
                              drop_last   = False,
                              pin_memory  = False)

    # **************************** Model  ***********************
    logger.info("initializing model")
    if config['resume']: 
        with open(config['lab_dir'], 'r') as f:
            config['label_to_id'] = load(f) 

    model = BertFine.from_pretrained(config['bert_model_dir'],
                                     cache_dir=config['cache_dir'],
                                     num_classes = len(config['label_to_id']))


    # ************************** Optimizer  *************************
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(
        len(train_dataset.examples) / config['batch_size'] / config['gradient_accumulation_steps'] * config['epochs'])
    # t_total: total number of training steps for the learning rate schedule
    # warmup: portion of t_total for the warmup
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = config['learning_rate'],
                         warmup = config['warmup_proportion'],
                         t_total = num_train_steps)

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # save model
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['checkpoint_dir'],
                                       mode             = config['mode'],
                                       monitor          = config['monitor'],
                                       save_best_only   = config['save_best_only'],
                                       best_model_name  = config['best_model_name'],
                                       epoch_model_name = config['epoch_model_name'],
                                       arch             = config['arch'],
                                       logger           = logger)
    # monitor
    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = config['arch'])

    # TensorBoard
    start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    writer_dir = os.path.join(config['writer_dir'], config['feature-based'], start_time)
    TSBoard = WriterTensorboardX(writer_dir = writer_dir,
                                 logger     = logger,
                                 enable     = True)
    # learning rate
    lr_scheduler = BertLr(optimizer = optimizer,
                          lr        = config['learning_rate'],
                          t_total   = num_train_steps,
                          warmup    = config['warmup_proportion'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      train_data       = train_loader,
                      val_data         = valid_loader,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      criterion        = BCEWithLogLoss(),
                      logger           = logger,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      TSBoard          = TSBoard,
                      resume           = config['resume'],
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      label_to_id      = config['label_to_id'],
                      evaluate_auc     = AUC(sigmoid=True),
                      evaluate_f1      = F1Score(sigmoid=True),
                      incorrect        = Incorrect(sigmoid=True))

    trainer.summary()
    trainer.train()

    # release cache
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
