import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseRecommender

class BPRMF(BaseRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.num_users = config['data']['num_users']
        self.num_items = config['data']['num_items']
        self.embedding_dim = config['models']['bprmf']['embedding_dim']
        self.reg_weight = config['models']['bprmf']['reg_weight']
        
        # Initialize user and item embeddings
        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embeddings(user_ids)
        item_embed = self.item_embeddings(item_ids)
        return torch.sum(user_embed * item_embed, dim=1)

    def calculate_loss(self, batch):
        users = batch['user']
        pos_items = batch['pos_item']
        neg_items = batch['neg_items']
        
        # Get embeddings
        user_embed = self.user_embeddings(users)
        pos_embed = self.item_embeddings(pos_items)
        neg_embed = self.item_embeddings(neg_items)
        
        # Calculate positive and negative scores
        pos_scores = torch.sum(user_embed * pos_embed, dim=1, keepdim=True)
        neg_scores = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.sum(user_embed ** 2) +
            torch.sum(pos_embed ** 2) +
            torch.sum(neg_embed ** 2)
        )
        
        return loss + l2_loss

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def get_user_embedding(self, user_id):
        return self.user_embeddings(user_id)

    def get_item_embedding(self, item_id):
        return self.item_embeddings(item_id)