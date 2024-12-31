import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
from spektral.layers import GCNConv
import networkx as nx

class MovieGraphOptimizer:
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
        
        # 计算电影之间的相似度
        movie_similarity = user_movie_matrix.T @ user_movie_matrix
        
        # 构建图
        self.graph = nx.Graph()
        
        # 添加节点
        for movie_id in movies:
            self.graph.add_node(movie_id)
            
        # 添加边（基于相似度）
        rows, cols = movie_similarity.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # 避免重复边
                similarity = movie_similarity[i, j]
                if similarity > self.config.SIMILARITY_THRESHOLD:
                    self.graph.add_edge(
                        movies[i], 
                        movies[j], 
                        weight=float(similarity)
                    )
        
        # 构建邻接矩阵
        self.adjacency_matrix = nx.adjacency_matrix(self.graph)
        
        return self.graph
        
    def create_node_features(self, movies_df, tag_features_df):
        """创建节点特征矩阵"""
        # 合并电影特征和标签特征
        movie_features = movies_df.merge(
            tag_features_df,
            on='movieId',
            how='left'
        )
        
        # 填充缺失值
        tag_columns = [f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]
        movie_features[tag_columns] = movie_features[tag_columns].fillna(0)
        
        self.node_features = movie_features[tag_columns].values
        return self.node_features

class GraphEnhancedDeepFM(tf.keras.Model):
    def __init__(self, base_model, graph_optimizer, config):
        super(GraphEnhancedDeepFM, self).__init__()
        self.base_model = base_model
        self.graph_optimizer = graph_optimizer
        self.config = config
        
        # 图卷积层
        self.gcn_layers = [
            GCNConv(config.GCN_UNITS)
            for _ in range(config.GCN_LAYERS)
        ]
        
        # 融合层
        self.fusion_layer = tf.keras.layers.Dense(
            config.FUSION_UNITS,
            activation='relu'
        )
        
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # 获取基础DeepFM的输出
        base_output = self.base_model(inputs)
        
        # 图卷积处理
        x = self.node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer([x, self.adjacency_matrix])
        
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

# 配置类扩展
class GraphConfig:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.3
        self.GCN_UNITS = 64
        self.GCN_LAYERS = 2
        self.FUSION_UNITS = 32