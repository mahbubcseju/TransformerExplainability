import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class TransformerExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def explain(self, input_ids, labels, steps=20, start_layer=4):
        self.model.eval()

        input = self.model.encoder.roberta.embeddings(input_ids)
        output, _, attention = self.model(input_ids=input_ids, inputs_embeds=input, explain=True)

        b = input.shape[0]
        b, h, s, _ = attention[-1].shape
        num_blocks = len(attention)

        states = attention[-1].mean(1)[:, 0, :].reshape(b, 1, s)
        print(states.shape)
        for i in range(start_layer, num_blocks - 1)[::-1]:
            attn = attention[i].mean(1)
            states_ = states
            states = states.bmm(attn)
            states += states_
            print(states.shape)

        total_gradients = torch.zeros(b, h, s, s).cpu()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha
            # backward propagation
            output, _, attention = self.model(input_ids=input_ids, inputs_embeds=data_scaled, explain=True)
            one_hot = np.zeros((b, 2), dtype=np.float32)
            one_hot[np.arange(b), labels] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cpu() * output)
            # Gradient passing
            self.model.zero_grad()
            attention[-1].retain_grad()
            one_hot.backward(retain_graph=True)
            # cal grad
            gradients = attention[-1].grad
            total_gradients += gradients

        W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        states = states * W_state
        return states[:, 0, :]

    def create_explanations(self, dataloader):
        result = []
        for id, batch in enumerate(dataloader):
            input_ids, labels, idx = batch
            scores = self.explain(input_ids, labels).cpu().detach().numpy().tolist()
            tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            result.append({
                'tokens': tokens,
                'scores': scores,
                'id': idx[0],

            })
            print("Explanation done with  id: ", id)
            break
        return result
