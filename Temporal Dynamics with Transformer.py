import tensorflow as tf
import numpy as np
from datetime import datetime

class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        
        # 时间编码层
        self.time_embedding = tf.keras.layers.Embedding(
            input_dim=config.MAX_TIME_STEPS,
            output_dim=config.TIME_EMBEDDING_DIM
        )
        
        # Multi-head attention layers
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(
                num_heads=config.NUM_HEADS,
                key_dim=config.TIME_EMBEDDING_DIM,
                value_dim=config.TIME_EMBEDDING_DIM
            ) for _ in range(config.NUM_TRANSFORMER_LAYERS)
        ]
        
        # Feed-forward networks
        self.ffn_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(config.FFN_UNITS, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(config.TIME_EMBEDDING_DIM)
            ]) for _ in range(config.NUM_TRANSFORMER_LAYERS)
        ]
        
        # Layer normalization
        self.ln_layers = [
            tf.keras.layers.LayerNormalization()
            for _ in range(config.NUM_TRANSFORMER_LAYERS * 2)
        ]
        
    def create_time_series_mask(self, length):
        """创建用于时间序列的mask"""
        mask = 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)
        return mask
    
    def call(self, inputs, training=False):
        # 提取时间序列特征
        time_features = self.time_embedding(inputs['time_idx'])
        
        # 创建attention mask
        mask = self.create_time_series_mask(tf.shape(time_features)[1])
        
        # Transformer编码器层
        x = time_features
        for i in range(self.config.NUM_TRANSFORMER_LAYERS):
            # Multi-head attention
            attn_output = self.attention_layers[i](
                query=x,
                key=x,
                value=x,
                attention_mask=mask,
                training=training
            )
            x = self.ln_layers[i*2](x + attn_output)
            
            # Feed-forward network
            ffn_output = self.ffn_layers[i](x)
            x = self.ln_layers[i*2+1](x + ffn_output)
        
        return x

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def extract_temporal_features(self, ratings_df):
        """提取时间序列特征"""
        # 按电影和时间排序
        ratings_df = ratings_df.sort_values(['movieId', 'timestamp'])
        
        # 为每部电影创建时间窗口
        movie_temporal_features = []
        
        for movie_id, group in ratings_df.groupby('movieId'):
            # 计算时间窗口
            timestamps = group['timestamp'].values
            ratings = group['rating'].values
            
            # 创建固定长度的时间序列
            max_len = self.config.MAX_TIME_STEPS
            if len(timestamps) > max_len:
                # 如果序列太长，进行采样
                indices = np.linspace(0, len(timestamps)-1, max_len, dtype=int)
                timestamps = timestamps[indices]
                ratings = ratings[indices]
            else:
                # 如果序列太短，进行填充
                pad_length = max_len - len(timestamps)
                timestamps = np.pad(timestamps, (0, pad_length), 'constant')
                ratings = np.pad(ratings, (0, pad_length), 'constant')
            
            movie_temporal_features.append({
                'movieId': movie_id,
                'timestamps': timestamps,
                'ratings': ratings
            })
        
        return movie_temporal_features

class Config:
    def __init__(self):
        self.MAX_TIME_STEPS = 50
        self.TIME_EMBEDDING_DIM = 32
        self.NUM_HEADS = 4
        self.NUM_TRANSFORMER_LAYERS = 2
        self.FFN_UNITS = 64
        
def create_temporal_dataset(temporal_features, config):
    """创建用于训练的时间序列数据集"""
    features = {
        'movieId': [],
        'time_idx': [],
        'timestamps': [],
        'ratings': []
    }
    
    for item in temporal_features:
        features['movieId'].append(item['movieId'])
        features['timestamps'].append(item['timestamps'])
        features['ratings'].append(item['ratings'])
        features['time_idx'].append(np.arange(config.MAX_TIME_STEPS))
    
    # 转换为tensorflow数据集
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset