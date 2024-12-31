#如果评分矩阵非常稀疏，冷启动问题仍然突出，我们可以显式地引入低秩优化方法（如矩阵分解）来填补稀疏矩阵中的空缺评分。
from sklearn.decomposition import TruncatedSVD

# 构造评分矩阵
rating_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# 使用SVD分解
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(rating_matrix)
item_factors = svd.components_.T

# 添加到原始特征
ratings_df['user_svd_feature'] = user_factors[ratings_df['userId']]
ratings_df['item_svd_feature'] = item_factors[ratings_df['movieId']]
