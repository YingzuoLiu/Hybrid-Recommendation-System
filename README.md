# Hybrid-Recommendation-System
Dataset : https://grouplens.org/datasets/movielens/

### 项目总结：推荐系统的特点与实现逻辑

#### 1. 背景与问题：特征不足  
**问题**：  
在使用 **MovieLens 32M** 数据集进行推荐系统开发时，存在以下挑战：  
- 电影信息和用户评分数据缺乏足够的特征，无法充分捕获复杂的用户偏好。  
- 标签数据噪音较大且存在冗余，无法直接用于模型训练。  

**解决方案**：  
- **时间特征构建**：通过`timestamp`字段提取时间相关特征（小时、星期几、月份、年份）。  
- **标签处理**：对标签数据进行清洗，使用TF-IDF方法对电影标签文本进行向量化，生成固定维度的标签特征（如100维）。  
- **特征填充**：对标签特征和缺失的ID特征进行填充，保证模型输入完整性。

---

#### 2. 模型结构与训练方法：Embedding与Chunk训练  
为解决特征稀疏问题，采用**DeepFM模型**，将特征嵌入到低维空间进行训练：  
- **Embedding层**：  
  对分类特征（`userId`、`movieId`等）和数值特征进行嵌入处理。标签特征经过Dense层进行非线性映射。  
- **FM层**：  
  特征交互部分通过 **因子分解机（Factorization Machine）** 捕获特征之间的二阶交互效果。  
- **Deep层**：  
  构建了一个多层神经网络（512 → 256 → 128），用于学习特征的深度非线性组合。  

**Chunk训练**：  
由于数据量较大，采用分批加载和处理数据（如`chunksize=50,000`），避免内存溢出。

---

#### 3. 网络调优：Deep层优化  
在Deep层结构中，通过以下方法提升模型性能：  
- **激活函数**：每一层Dense网络使用`ReLU`激活函数，提高非线性表达能力。  
- **Batch Normalization**：对每一层的输出进行归一化，加速模型收敛并减少过拟合风险。  
- **Dropout**：设置30%的Dropout比例，防止过拟合。  

---

#### 4. 训练过程中的正则化与技巧  
- **L2正则化**：对Embedding层的权重施加L2正则化，抑制过大的权重。  
- **学习率调整**：通过 `ReduceLROnPlateau` 回调函数，根据验证集的`loss`动态调整学习率。  
- **早停机制**：使用 `EarlyStopping` 防止模型在验证集上过拟合。

---

#### 5. 结果与评估  
在训练完成后，模型在测试集上进行了评估，结果如下：  
- **均方误差（MSE）**：模型在评分预测上表现良好，测试集误差较低。  
- **平均绝对误差（MAE）**：进一步验证模型预测的准确性。  

**总结**：  
通过对特征工程的改进、DeepFM模型的使用以及网络结构的优化，成功构建了一个性能优异的推荐系统。特征交互与深度学习结合使得模型能够更好地捕捉用户和电影之间的复杂关系。  

---

### 下一步工作  
- 引入更多外部特征（如用户观影时长、地理位置信息）。  
- 探索**Transformer结构**在推荐系统中的应用，进一步提升推荐效果。  
- 对推荐结果进行用户实验，验证其实际可用性和满意度。  
