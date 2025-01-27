import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

import numpy as np
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, user_items_dict, num_items, num_users, negative_sample_size=4):
        self.user_items_dict = user_items_dict
        self.num_items = num_items
        self.negative_sample_size = negative_sample_size
        self.users = list(user_items_dict.keys())
        
        # Check user ID range
        self.max_user_id = max(self.users)
        assert self.max_user_id < num_users, f"User ID {self.max_user_id} out of range (max allowed: {num_users-1})"

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = self.user_items_dict[user]
        
        # Sample positive item
        pos_item = np.random.choice(pos_items)
        
        # Sample negative items
        neg_items = []
        while len(neg_items) < self.negative_sample_size:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in pos_items:
                neg_items.append(neg_item)
        
        return {
            'user': torch.LongTensor([user]),
            'pos_item': torch.LongTensor([pos_item]),
            'neg_items': torch.LongTensor(neg_items)
        }
    
class DataManager:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        # 加载评分数据
        ratings_df = pd.read_csv(f"{self.config['data']['dataset_path']}/ratings.csv")
        
        # 创建从0开始的连续用户和物品ID映射
        unique_user_ids = ratings_df['userId'].unique()
        unique_item_ids = ratings_df['movieId'].unique()
        
        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_user_ids)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(unique_item_ids)}
        
        # 更新配置中的用户和物品数量
        self.config['data']['num_users'] = len(self.user_mapping)
        self.config['data']['num_items'] = len(self.item_mapping)
        
        # 转换ID为索引
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_mapping)
        ratings_df['item_idx'] = ratings_df['movieId'].map(self.item_mapping)
        
        # 创建用户-物品字典
        user_items_dict = {}
        for user, items in ratings_df.groupby('user_idx')['item_idx']:
            user_items_dict[user] = items.values
            
        return self.create_train_val_test_split(user_items_dict)
    
    def create_train_val_test_split(self, user_items_dict):
        users = list(user_items_dict.keys())
        train_users, test_users = train_test_split(
            users, 
            test_size=self.config['data']['test_ratio'],
            random_state=self.config['training']['seed']
        )
        
        train_users, val_users = train_test_split(
            train_users,
            test_size=self.config['data']['val_ratio']/(1-self.config['data']['test_ratio']),
            random_state=self.config['training']['seed']
        )
        
        splits = {
            'train': {u: user_items_dict[u] for u in train_users},
            'val': {u: user_items_dict[u] for u in val_users},
            'test': {u: user_items_dict[u] for u in test_users}
        }
        
        return splits