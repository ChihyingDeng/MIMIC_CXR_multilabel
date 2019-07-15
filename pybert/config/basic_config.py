#encoding:utf-8
from os import path
import multiprocessing


BASE_DIR = 'pybert'

configs = {
    'arch':'bert-multi-label',
    'raw_data_path': path.sep.join([BASE_DIR,'dataset/raw/position.csv']),
    'train_file_path': path.sep.join([BASE_DIR,'dataset/processed/train.csv']),
    'valid_file_path': path.sep.join([BASE_DIR,'dataset/processed/valid.csv']),

    'lab_dir': path.sep.join([BASE_DIR,'dataset/raw/label.json']),
    'result_dir':path.sep.join([BASE_DIR, 'output/result']), #output path
    'raw_reports_dir':path.sep.join([BASE_DIR, 'dataset/report']), #input report
    
    'log_dir': path.sep.join([BASE_DIR, 'output/log']), 
    'writer_dir': path.sep.join([BASE_DIR, 'output/TSboard']), #TSboard
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']), 
    'checkpoint_dir': path.sep.join([BASE_DIR, 'output/checkpoints']), #model
    'cache_dir': path.sep.join([BASE_DIR,'model/']),
    'misclassified_dir': path.sep.join([BASE_DIR, 'output/misclassified']), 

    # pretrained Bert model
    'bert_model_dir': path.sep.join([BASE_DIR, 'model/pretrain/bert_large/']),

    'valid_size': 0, 
    'max_seq_len': 256,  
    'do_lower_case':True,
    'batch_size': 16,   # how many samples to process at once
    'epochs': 30,       # number of epochs to train
    'start_epoch': 1,
    'warmup_proportion': 0.1, # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    'gradient_accumulation_steps':1, # Number of updates steps to accumulate before performing a backward/update pass.
    'learning_rate': 2e-5,
    'n_gpus': [0], 

    'num_workers': multiprocessing.cpu_count(), 
    'resume': False,
    'shuffle': True,
    'multiclass': False, # the label is more than two classes (0 and 1)
    'seed': 500,
    'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'val_loss',  
    'early_patience': 10,   # early_stopping
    'save_best_only': True, 
    'best_model_name': '{arch}-best-position.pth', 
    'epoch_model_name': '{arch}-{epoch}-{val_loss}.pth', 
    'save_checkpoint_freq': 10, 
    'print_incorrect': True,
    'feature-based': 'Concat_Last_Four' # Finetune_All, First, Second_to_Last, 
                               # Last, Sum_Last_Four, Concat_Last_Four, Sum_All
}

