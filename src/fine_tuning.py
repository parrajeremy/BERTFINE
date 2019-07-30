import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert import BertAdam, WarmupLinearSchedule

from src.training import BERT_MODEL, CACHE_DIR, num_labels, device, optimizer_grouped_parameters, LEARNING_RATE,\
    WARMUP_PROPORTION, num_train_optimization_steps

if __name__=="__main__":
    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
    # model = BertForSequenceClassification.from_pretrained(CACHE_DIR + 'cased_base_bert_pytorch.tar.gz', cache_dir=CACHE_DIR, num_labels=num_labels)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=LEARNING_RATE,
                         warmup=WARMUP_PROPORTION,
                         t_total=num_train_optimization_steps)