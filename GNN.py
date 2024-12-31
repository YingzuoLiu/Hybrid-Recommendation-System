import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from tensorflow.keras import layers

class GNNLayer(layers.Layer):
    """自定义GNN层"""
    def __init__(self, units, activation='relu', aggregation_type='mean', **kwargs):
        super(GNNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.aggregation_type = aggregation_type
        
        # 转换矩阵
        self.transform = layers.Dense(units, use_bias=False)
        # 更新矩阵
        self.update = layers.Dense(units)
        
    def aggregate_neighbors(self, node_features, adjacency_matrix):
        """聚合邻居节点信息"""
        # 确保邻接矩阵是稀疏tensor
        if not isinstance(adjacency_matrix, tf.sparse.SparseTensor):
            adjacency_matrix = tf.sparse.from_dense(adjacency_matrix)
            
        # 不同的聚合方式
        if self.aggregation_type == 'mean':
            # 计算度矩阵的逆
            degree = tf.sparse.reduce_sum(adjacency_matrix, axis=1)
            degree_inv = tf.pow(degree, -1)
            degree_inv = tf.where(tf.math.is_inf(degree_inv), 0., degree_inv)
            degree_inv = tf.sparse.from_dense(tf.linalg.diag(degree_inv))
            
            # 归一化邻接矩阵
            norm_adj = tf.sparse.sparse_dense_matmul(degree_inv, adjacency_matrix)
            
            # 聚合邻居特征
            neighbor_features = tf.sparse.sparse_dense_matmul(norm_adj, node_features)
        elif self.aggregation_type == 'sum':
            neighbor_features = tf.sparse.sparse_dense_matmul(adjacency_matrix, node_features)
        elif self.aggregation_type == 'max':
            # 实现最大池化聚合
            neighbor_features = tf.sparse.sparse_dense_matmul(adjacency_matrix, node_features)
            neighbor_features = tf.reduce_max(
                tf.reshape(neighbor_features, [-1, self.units]), 
                axis=0
            )
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation_type}")
            
        return neighbor_features
        
    def call(self, inputs):
        node_features, adjacency_matrix = inputs
        
        # 转换节点特征
        transformed_features = self.transform(node_features)
        
        # 聚合邻居信息
        aggregated_features = self.aggregate_neighbors(transformed_features, adjacency_matrix)
        
        # 更新节点特征
        updated_features = self.update(
            tf.concat([node_features, aggregated_features], axis=-1)
        )
        
        return self.activation(updated_features)

class MovieGNN:
    def __init__(self, config):
        self.config = config
        self.graph = None
        self.adjacency_matrix = None
        self.node_features = None
        
    def build_movie_graph(self, ratings_df, movies_df):
        """构建电影关系图"""
        # 创建用户-电影评分矩阵
        users = ratings_df['userId'].unique()
        movies = ratings_df['movieId'].unique()
        
        user_movie_matrix = csr_matrix(
            (ratings_df['rating'],
             (ratings_df['userId'].astype('category').cat.codes,
              ratings_df['movieId'].astype('category').cat.codes))
        )
        
        # 计算电影之间的相似度（考虑多种关系）
        # 1. 基于用户行为的相似度
        behavior_similarity = user_movie_matrix.T @ user_movie_matrix
        
        # 2. 基于电影特征的相似度
        if 'genres' in movies_df.columns:
            genres = movies_df['genres'].str.get_dummies(sep='|')
            genre_similarity = genres @ genres.T
        else:
            genre_similarity = None
        
        # 构建图
        self.graph = nx.Graph()
        
        # 添加节点
        for movie_id in movies:
            self.graph.add_node(movie_id)
            
        # 添加边（结合多种相似度）
        rows, cols = behavior_similarity.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # 避免重复边
                similarity = behavior_similarity[i, j]
                if genre_similarity is not None:
                    # 结合两种相似度
                    similarity = 0.7 * similarity + 0.3 * genre_similarity.iloc[i, j]
                
                if similarity > self.config.SIMILARITY_THRESHOLD:
                    self.graph.add_edge(
                        movies[i], 
                        movies[j], 
                        weight=float(similarity)
                    )
        
        # 构建邻接矩阵
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        return self.graph
        
    def create_node_features(self, movies_df, tag_features_df):
        """创建节点特征矩阵"""
        movie_features = movies_df.merge(
            tag_features_df,
            on='movieId',
            how='left'
        )
        
        # 标签特征
        tag_columns = [f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]
        movie_features[tag_columns] = movie_features[tag_columns].fillna(0)
        
        # 合并其他可能的特征（如类型、年份等）
        feature_cols = tag_columns
        if 'genres' in movies_df.columns:
            genre_features = movies_df['genres'].str.get_dummies(sep='|')
            movie_features = movie_features.join(genre_features)
            feature_cols.extend(genre_features.columns)
            
        self.node_features = movie_features[feature_cols].values
        return self.node_features

class GNNEnhancedDeepFM(tf.keras.Model):
    def __init__(self, base_model, movie_gnn, config):
        super(GNNEnhancedDeepFM, self).__init__()
        self.base_model = base_model
        self.movie_gnn = movie_gnn
        self.config = config
        
        # GNN层
        self.gnn_layers = [
            GNNLayer(
                units=config.GNN_HIDDEN_UNITS[i],
                aggregation_type=config.GNN_AGGREGATION_TYPE
            )
            for i in range(len(config.GNN_HIDDEN_UNITS))
        ]
        
        # 融合层
        self.fusion_layer = tf.keras.layers.Dense(
            config.FUSION_UNITS,
            activation='relu'
        )
        
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # 基础DeepFM输出
        base_output = self.base_model(inputs)
        
        # GNN特征提取
        x = self.node_features
        adj = self.adjacency_matrix
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer([x, adj])
        
        # 获取当前batch的电影对应的图特征
        movie_indices = tf.cast(
            inputs['movieId'] * (self.node_features.shape[0] - 1),
            tf.int32
        )
        graph_features = tf.gather(x, movie_indices)
        
        # 特征融合
        combined_features = tf.concat([base_output, graph_features], axis=1)
        fusion_output = self.fusion_layer(combined_features)
        
        # 最终预测
        final_output = self.output_layer(fusion_output)
        return final_output

class GNNConfig:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.3
        self.GNN_HIDDEN_UNITS = [64, 32]  # 每层的隐藏单元数
        self.GNN_AGGREGATION_TYPE = 'mean'  # 'mean', 'sum', 'max'
        self.FUSION_UNITS = 32