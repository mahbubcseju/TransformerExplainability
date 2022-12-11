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
        # self.softmax = torch.softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
        
    def forward(self, input_ids=None,labels=None, explain=False): 
        outputs=self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits=outputs[0]
        prob=torch.softmax(logits, dim=-1)
        if labels is not None:
            if explain:
                return self.criterion(prob, labels), prob, outputs[1], outputs[2]
            return self.criterion(prob, labels), prob
        else:
            if explain:
                return  prob, outputs[1], outputs[2]
            return prob
