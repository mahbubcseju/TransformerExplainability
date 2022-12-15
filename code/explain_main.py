# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

from transformer_explainer import TransformerExplainer
from model import Model

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, js[args.idx_key], js['target'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), self.examples[i].idx


class CodebertModel:
    """Simple sentiment analysis model."""
    def __init__(self):
        self.config_class = RobertaConfig
        self.model_class = RobertaForSequenceClassification
        self.tokenizer_class = RobertaTokenizer
        self.model_name_or_path = 'microsoft/codebert-base'
        self.model_config = self.config_class.from_pretrained(self.model_name_or_path,
                num_labels=2,
                output_hidden_states=True,
                output_attentions=True)
        self.tokenizer = self.tokenizer_class.from_pretrained('microsoft/codebert-base')

        model = self.model_class.from_pretrained(self.model_name_or_path,
                                                from_tf=bool('.ckpt' in self.model_name_or_path),
                                                 config=self.model_config)
        self.model = Model(model, self.model_config,  self.tokenizer)
        self.output_dir="./saved_models"

    def activate_evaluation(self):
        self.model.eval()

    def load_model(self, args):
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        self.model.load_state_dict(torch.load(output_dir, map_location=args.device))
        self.model.to(args.device)


    def test(self, args):
        model = self.model
        tokenizer = self.tokenizer
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_dataset = TextDataset(tokenizer, args,args.test_data_file)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running Test *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        model.eval()
        logits=[]   
        labels=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            inputs = batch[0].to(args.device)        
            label=batch[1].to(args.device) 
            with torch.no_grad():
                logit = model(inputs)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits=np.concatenate(logits, 0)
        labels=np.concatenate(labels, 0)

        acts = np.array([0 if (sample[0] > sample[1]) else 1 for sample in labels])
        preds = np.array([0 if (sample[0] > sample[1]) else 1 for sample in logits])
        eval_acc=np.mean(acts==preds)
        print("Accuracy: ", eval_acc)
    
    def create_explanations(self, args):
        eval_dataset = TextDataset(self.tokenizer, args, args.test_data_file)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        explainer = TransformerExplainer(self.model, self.tokenizer)
        return explainer.create_explanations(eval_dataloader)
 
