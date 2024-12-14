import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import logging
import re
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.RATINGS_PATH = 'ratings.csv'
        self.MOVIES_PATH = 'movies.csv'
        self.TAGS_PATH = 'tags.csv'
        self.LINKS_PATH = 'links.csv'
        self.EMBEDDING_DIM = 16  # 增加embedding维度
        self.DEEP_LAYERS = [512, 256, 128]  # 加深网络
        self.BATCH_SIZE = 4096
        self.EPOCHS = 10
        self.LEARNING_RATE = 0.001
        self.NUMERICAL_FEATURES = ['timestamp', 'hour', 'day_of_week', 'month', 'year']
        self.CATEGORICAL_FEATURES = ['userId', 'movieId', 'imdbId', 'tmdbId']
        self.TAG_FEATURES_DIM = 100
        self.LABEL_NAME = 'rating'
        self.CHUNK_SIZE = 50000
        self.MAX_SAMPLES = 1000000
        self.MIN_TAG_LENGTH = 2
        self.MAX_TAG_LENGTH = 50

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scalers = {}
        self.tfidf = TfidfVectorizer(
            max_features=config.TAG_FEATURES_DIM,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
    
    def add_time_features(self, df):
        """添加时间相关特征"""
        dt = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['year'] = dt.dt.year
        return df
    
    def clean_tag(self, tag):
        tag = str(tag).lower().strip()
        if tag.isdigit() or not re.search('[a-zA-Z]', tag):
            return ''
        tag = re.sub(r'[^a-zA-Z\s]', '', tag)
        if len(tag) < self.config.MIN_TAG_LENGTH or len(tag) > self.config.MAX_TAG_LENGTH:
            return ''
        return tag.strip()
        
    def process_tags(self, tags_df):
        logger.info("开始处理标签数据...")
        tags_df['clean_tag'] = tags_df['tag'].apply(self.clean_tag)
        tags_df = tags_df[tags_df['clean_tag'] != '']
        
        total_tags = len(tags_df)
        unique_tags = len(tags_df['clean_tag'].unique())
        logger.info(f"清理后共有 {total_tags} 个标签，其中唯一标签 {unique_tags} 个")
        
        if total_tags == 0:
            empty_features = pd.DataFrame(
                columns=[f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]
            )
            empty_features['movieId'] = tags_df['movieId'].unique()
            return empty_features
        
        tags_grouped = tags_df.groupby('movieId')['clean_tag'].agg(lambda x: ' '.join(x)).reset_index()
        tag_features = self.tfidf.fit_transform(tags_grouped['clean_tag'])
        
        tag_features_df = pd.DataFrame(
            tag_features.toarray(),
            columns=[f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]
        )
        tag_features_df['movieId'] = tags_grouped['movieId']
        
        if hasattr(self.tfidf, 'get_feature_names_out'):
            feature_names = self.tfidf.get_feature_names_out()
        else:
            feature_names = self.tfidf.get_feature_names()
        
        logger.info("最常见的10个标签:")
        for idx in tag_features.sum(axis=0).argsort()[0, -10:].tolist()[0]:
            logger.info(f"- {feature_names[idx]}")
            
        return tag_features_df

    def load_data(self):
        logger.info("加载数据...")
        chunks = []
        total_rows = 0
        for chunk in pd.read_csv(self.config.RATINGS_PATH, encoding='latin1', chunksize=self.config.CHUNK_SIZE):
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows >= self.config.MAX_SAMPLES:
                break
        
        ratings_df = pd.concat(chunks, ignore_index=True)
        if total_rows > self.config.MAX_SAMPLES:
            ratings_df = ratings_df.head(self.config.MAX_SAMPLES)
        
        # 添加时间特征
        ratings_df = self.add_time_features(ratings_df)
        
        movies_df = pd.read_csv(self.config.MOVIES_PATH, encoding='latin1')
        tags_df = pd.read_csv(self.config.TAGS_PATH, encoding='latin1')
        links_df = pd.read_csv(self.config.LINKS_PATH, encoding='latin1')
        
        ratings_df['movieId'] = ratings_df['movieId'].astype(int)
        tags_df['movieId'] = tags_df['movieId'].astype(int)
        links_df['movieId'] = links_df['movieId'].astype(int)
        
        tag_features_df = self.process_tags(tags_df)
        
        df = ratings_df.merge(links_df, on='movieId', how='left')
        df = df.merge(tag_features_df, on='movieId', how='left')
        
        tag_columns = [f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]
        df[tag_columns] = df[tag_columns].fillna(0)
        df[['imdbId', 'tmdbId']] = df[['imdbId', 'tmdbId']].fillna(-1)
        
        logger.info(f"总共加载 {len(df)} 条数据，包含 {len(tag_columns)} 个标签特征")
        return df

    def preprocess_features(self, df):
        df = df.copy()
        df[self.config.LABEL_NAME] = (df[self.config.LABEL_NAME] - 1) / 4

        for feat in self.config.CATEGORICAL_FEATURES:
            if feat not in self.label_encoders:
                self.label_encoders[feat] = LabelEncoder()
                df[feat] = self.label_encoders[feat].fit_transform(df[feat].astype(str))
                df[feat] = df[feat] / (len(self.label_encoders[feat].classes_) - 1)
            else:
                df[feat] = self.label_encoders[feat].transform(df[feat].astype(str))
                df[feat] = df[feat] / (len(self.label_encoders[feat].classes_) - 1)

        for feat in self.config.NUMERICAL_FEATURES:
            if feat not in self.scalers:
                self.scalers[feat] = StandardScaler()
                df[feat] = self.scalers[feat].fit_transform(df[feat].values.reshape(-1, 1)).ravel()
            else:
                df[feat] = self.scalers[feat].transform(df[feat].values.reshape(-1, 1)).ravel()

        return df

    def create_dataset(self, df, shuffle=True):
        features = {}
        
        for col in self.config.CATEGORICAL_FEATURES:
            features[col] = tf.cast(df[col].values, tf.float32)
            
        for col in self.config.NUMERICAL_FEATURES:
            features[col] = tf.cast(df[col].values, tf.float32)
            
        tag_features = tf.cast(
            df[[f'tag_feature_{i}' for i in range(self.config.TAG_FEATURES_DIM)]].values,
            tf.float32
        )
        features['tag_features'] = tag_features

        labels = tf.cast(df[self.config.LABEL_NAME].values, tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(df), 10000))
        dataset = dataset.batch(self.config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return dataset

class DeepFMModel(tf.keras.Model):
    def __init__(self, feature_dims, config):
        super(DeepFMModel, self).__init__()
        self.config = config

        # Embedding层
        self.embedding_layers = {
            feat: tf.keras.layers.Embedding(
                input_dim=dim,
                output_dim=config.EMBEDDING_DIM,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
            )
            for feat, dim in feature_dims.items()
        }

        # 数值特征处理层
        self.numeric_layers = {
            feat: tf.keras.layers.Dense(config.EMBEDDING_DIM)
            for feat in config.NUMERICAL_FEATURES
        }

        # 标签特征处理层
        self.tag_dense = tf.keras.layers.Dense(config.EMBEDDING_DIM, activation='relu')
        
        # Deep部分
        self.deep_layers = []
        for units in config.DEEP_LAYERS:
            self.deep_layers.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3)
            ])

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # First-order部分
        first_order = []
        for feat in self.config.CATEGORICAL_FEATURES:
            first_order.append(self.embedding_layers[feat](inputs[feat]))
        
        # 处理数值特征
        for feat in self.config.NUMERICAL_FEATURES:
            numeric_embedding = self.numeric_layers[feat](tf.expand_dims(inputs[feat], -1))
            first_order.append(numeric_embedding)
        
        # 处理标签特征
        tag_embedding = self.tag_dense(inputs['tag_features'])
        first_order.append(tag_embedding)
        
        # FM部分 - 特征交互
        stack_embeddings = tf.stack(first_order, axis=1)
        sum_square = tf.square(tf.reduce_sum(stack_embeddings, axis=1))
        square_sum = tf.reduce_sum(tf.square(stack_embeddings), axis=1)
        fm_output = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
        
        # Deep部分
        deep_input = tf.concat(first_order, axis=1)
        for layer in self.deep_layers:
            deep_input = layer(deep_input)
        
        # 组合FM和Deep的输出
        final_output = self.output_layer(tf.concat([deep_input, fm_output], axis=1))
        return final_output

def main():
    config = Config()
    data_loader = DataLoader(config)

    logger.info("Loading and preprocessing data...")
    df = data_loader.load_data()
    df = data_loader.preprocess_features(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = data_loader.create_dataset(train_df)
    test_dataset = data_loader.create_dataset(test_df, shuffle=False)

    feature_dims = {feat: len(data_loader.label_encoders[feat].classes_) 
                   for feat in config.CATEGORICAL_FEATURES}
    model = DeepFMModel(feature_dims, config)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mse', 'mae']
    )

    logger.info("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config.EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
    )

    logger.info("Evaluating model...")
    test_loss, test_mse, test_mae = model.evaluate(test_dataset)
    logger.info(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

if __name__ == "__main__":
    main()