import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from models.base import BaseRecommender

class LightGCN(BaseRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.num_users = config['data']['num_users']
        self.num_items = config['data']['num_items']
        self.embedding_dim = config['models']['lightgcn']['embedding_dim']
        self.n_layers = config['models']['lightgcn']['n_layers']
        self.reg_weight = config['models']['lightgcn']['reg_weight']

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.adj_matrix = None
        self.norm_adj_matrix = None

    def create_adj_matrix(self, train_data):
        # Create adjacency matrix in COO format
        user_nodes, item_nodes = [], []
        for user, items in train_data.items():
            user_nodes.extend([user] * len(items))
            item_nodes.extend(items)

        # Create adjacency matrix
        adj_mat = sp.coo_matrix(
            ([1] * len(user_nodes), (user_nodes, item_nodes)),
            shape=(self.num_users, self.num_items)
        )

        # Create symmetric adjacency matrix
        adj_mat = sp.vstack([
            sp.hstack([sp.csr_matrix((self.num_users, self.num_users)), adj_mat]),
            sp.hstack([adj_mat.T, sp.csr_matrix((self.num_items, self.num_items))])
        ])

        # Convert to symmetric normalized adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        self.norm_adj_matrix = self._convert_sp_mat_to_tensor(norm_adj)

    def _convert_sp_mat_to_tensor(self, sp_mat):
        """Convert scipy sparse matrix to torch sparse tensor"""
        coo = sp_mat.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, users, items):
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        
        embeddings_list = [all_embeddings]
        
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix.to(all_embeddings.device),
                all_embeddings
            )
            embeddings_list.append(all_embeddings)
        
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.num_users, self.num_items]
        )

        user_embeddings = user_all_embeddings[users]
        item_embeddings = item_all_embeddings[items]
        
        return torch.sum(user_embeddings * item_embeddings, dim=1)

    def calculate_loss(self, batch):
        users = batch['user']
        pos_items = batch['pos_item']
        neg_items = batch['neg_items']

        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(self.user_embedding(users)) +
            torch.norm(self.item_embedding(pos_items)) +
            torch.norm(self.item_embedding(neg_items))
        )

        return loss + l2_loss

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def get_user_embedding(self, user_id):
        return self.user_embedding(user_id)

    def get_item_embedding(self, item_id):
        return self.item_embedding(item_id)