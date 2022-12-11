import torch

class Arg:
    def __init__(self):
        self.train_data_file = '../dataset/train.jsonl'
        self.device = torch.device('cpu')
        self.epoch = 5
        self.train_batch_size = 16
        self.adam_epsilon = 1e-8
        self.learning_rate = 2e-5
        self.max_grad_norm=1.0
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 1
        self.local_rank = -1
        self.output_dir = '/Users/mahbubcseju/Desktop/projects/TransformerExplainability/saved_models/func_jsonal/64'
        self.eval_data_file = '../dataset/valid.jsonl'
        self.eval_batch_size = 1
        self.evaluate_during_training = True
        self.test_data_file = '/Users/mahbubcseju/Desktop/projects/TransformerExplainability/data/func_jsonal/test.jsonl'
        self.idx_key='id'
        self.block_size=400