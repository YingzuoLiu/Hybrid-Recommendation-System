import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from models.base import BaseRecommender

class ItemKNN(BaseRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.k = config['models']['itemknn']['k']
        self.similarity_metric = config['models']['itemknn']['similarity_metric']
        self.num_users = config['data']['num_users']
        self.num_items = config['data']['num_items']
        self.similarity_matrix = None
        self.interaction_matrix = None

    def fit(self, train_data):
        # Convert training data to interaction matrix
        rows, cols, data = [], [], []
        for user, items in train_data.items():
            rows.extend([user] * len(items))
            cols.extend(items)
            data.extend([1] * len(items))
        
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(self.num_users, self.num_items)
        )
        
        # Calculate similarity matrix
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.interaction_matrix.T)
        else:
            raise NotImplementedError(f"Similarity metric {self.similarity_metric} not implemented")
        
        # Keep only top-k similar items
        for i in range(self.num_items):
            sim_row = self.similarity_matrix[i]
            top_k_idx = np.argpartition(sim_row, -self.k)[-self.k:]
            mask = np.zeros_like(sim_row, dtype=bool)
            mask[top_k_idx] = True
            sim_row[~mask] = 0
            self.similarity_matrix[i] = sim_row

    def predict(self, user_ids, item_ids):
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.cpu().numpy()
        if isinstance(item_ids, torch.Tensor):
            item_ids = item_ids.cpu().numpy()
        
        predictions = []
        for user, item in zip(user_ids, item_ids):
            user_interactions = self.interaction_matrix[user].toarray().flatten()
            item_similarities = self.similarity_matrix[item]
            
            # Calculate prediction score
            weighted_sum = np.sum(user_interactions * item_similarities)
            similarity_sum = np.sum(np.abs(item_similarities))
            
            prediction = weighted_sum / (similarity_sum + 1e-8)
            predictions.append(prediction)
            
        return torch.tensor(predictions)

    def calculate_loss(self, batch):
        raise NotImplementedError("ItemKNN is a memory-based method and doesn't require training")

    def get_user_embedding(self, user_id):
        return self.interaction_matrix[user_id].toarray().flatten()

    def get_item_embedding(self, item_id):
        return self.similarity_matrix[item_id]