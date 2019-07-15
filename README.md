<a href="https://doi.org/10.5281/zenodo.3336618"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3336618.svg" alt="DOI"></a>

# MIMIC_CXR_multilabel
1. [Structure of the code](README.md##Structure)
1. [Dependencies](README.md##Dependencies)

## Structure of the code

At the root of the project, you will see:

```text
├── pybert
|  ├── callback
|  |  ├── lrscheduler.py　　
|  |  ├── trainingmonitor.py　
|  |  └── ...
|  ├── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  ├── dataset　
|  |  ├── processed
|  |  └── raw
|  ├── io　　　　
|  |  ├── dataset.py　　
|  |  └── data_transformer.py　　
|  ├── model
|  |  ├── nn　
|  |  |  └── bert_fine.py
|  |  └── pretrain　
|  ├── output #save the ouput of model
|  ├── preprocessing #text preprocessing
|  |  ├── augmentation.py
|  |  └── preprocessor.py
|  ├── train #used for training a model
|  |  ├── losses.py
|  |  ├── metrics.py
|  |  ├── train_util.py
|  |  └── trainer.py 
|  └── utils # a set of utility functions
├── convert_tf_checkpoint_to_pytorch.py
├── train_bert_multi_label.py
```
## Dependencies
- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- pandas
- pytorch_pretrained_bert (load bert model)

