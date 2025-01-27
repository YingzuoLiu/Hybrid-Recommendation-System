import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseRecommender

class DeepFM(BaseRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.num_users = config['data']['num_users']
        self.num_items = config['data']['num_items']
        self.embedding_dim = config['models']['deepfm']['embedding_dim']
        self.mlp_dims = config['models']['deepfm']['mlp_dims']
        self.dropout = config['models']['deepfm']['dropout']
        
        # FM部分的embedding层
        self.fm_first_order_users = nn.Embedding(self.num_users, 1)
        self.fm_first_order_items = nn.Embedding(self.num_items, 1)
        self.fm_second_order_users = nn.Embedding(self.num_users, self.embedding_dim)
        self.fm_second_order_items = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Deep部分
        self.mlp_layers = nn.ModuleList()
        input_dim = self.embedding_dim * 2  # user和item embedding的拼接
        for dim in self.mlp_dims:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(self.dropout))
            input_dim = dim
        
        # 最终预测层
        self.predict_layer = nn.Linear(input_dim + 1, 1)  # +1 for FM output
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)

    def fm_layer(self, user_ids, item_ids):
        # First order
        first_order = self.fm_first_order_users(user_ids) + \
                     self.fm_first_order_items(item_ids)
        
        # Second order
        user_embed = self.fm_second_order_users(user_ids)
        item_embed = self.fm_second_order_items(item_ids)
        
        sum_square = torch.pow(user_embed + item_embed, 2)
        square_sum = torch.pow(user_embed, 2) + torch.pow(item_embed, 2)
        second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        return first_order + second_order

    def deep_layer(self, user_ids, item_ids):
        user_embed = self.fm_second_order_users(user_ids)
        item_embed = self.fm_second_order_items(item_ids)
        
        inputs = torch.cat([user_embed, item_embed], dim=1)
        
        for layer in self.mlp_layers:
            inputs = layer(inputs)
        
        return inputs

    def forward(self, user_ids, item_ids):
        fm_output = self.fm_layer(user_ids, item_ids)
        deep_output = self.deep_layer(user_ids, item_ids)
        
        output = self.predict_layer(torch.cat([deep_output, fm_output], dim=1))
        return output.squeeze(-1)

    def calculate_loss(self, batch):
        users = batch['user']
        pos_items = batch['pos_item']
        neg_items = batch['neg_items']
        
        batch_size = users.size(0)
        
        # Positive samples
        pos_scores = self.forward(users, pos_items)
        
        # Negative samples
        neg_scores = []
        for i in range(neg_items.size(1)):
            neg_score = self.forward(users, neg_items[:, i])
            neg_scores.append(neg_score)
        neg_scores = torch.stack(neg_scores, dim=1)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores)))
        
        return loss

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def get_user_embedding(self, user_id):
        return self.fm_second_order_users(user_id)

    def get_item_embedding(self, item_id):
        return self.fm_second_order_items(item_id)