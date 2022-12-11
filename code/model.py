# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.softmax = torch.softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits=outputs[0]
        prob=self.softmax(logits)
        if labels is not None:
            return self.criterion(prob, labels),prob
        else:
            return prob
