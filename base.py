import torch
import torch.nn as nn

class BaseRecommender(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def calculate_loss(self, batch):
        raise NotImplementedError
    
    def predict(self, user_ids, item_ids):
        raise NotImplementedError
    
    def get_user_embedding(self, user_id):
        raise NotImplementedError
    
    def get_item_embedding(self, item_id):
        raise NotImplementedError