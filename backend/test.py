import pandas as pd
import numpy as np
from pynndescent import NNDescent

# 示例数据，包含数组特征
data = {
    'scalar_feature1': [1.0, 2.0, 3.0],
    'scalar_feature2': [4.0, 5.0, 6.0],
    'array_feature': [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]  # 使用 numpy 数组
}

database_df = pd.DataFrame(data)


# 自定义距离函数
def compute_weighted_distance(row_a, row_b):
    # 计算标量特征的欧氏距离
    scalar_distance = np.linalg.norm(row_a[['scalar_feature1', 'scalar_feature2']].values -
                                     row_b[['scalar_feature1', 'scalar_feature2']].values)

    # 计算数组特征的 L1 距离
    hist1 = row_a['array_feature']
    hist2 = row_b['array_feature']
    vector_distance = np.sum(np.abs(hist1 - hist2))

    return scalar_distance + vector_distance


# 自定义距离函数用于 PyNNDescent
def custom_distance(a, b):
    return compute_weighted_distance(database_df.iloc[int(a)], database_df.iloc[int(b)])


# 使用 PyNNDescent 查找近邻
def knn(database_df, target_df):
    if target_df.shape[0] != 1:
        raise ValueError("target_df should contain exactly one row.")

    # 创建 NNDescent 实例
    nnd = NNDescent(database_df, metric=custom_distance)

    # 查询近邻
    indices, distances = nnd.query(target_df, k=2)  # k 是最近邻的数量

    # 输出结果
    nearest_neighbors = database_df.iloc[indices[0]]
    print(nearest_neighbors)


# 示例调用
target_data = {
    'scalar_feature1': [2.5],
    'scalar_feature2': [5.0],
    'array_feature': [np.array([3, 4])]  # 确保这里也使用 numpy 数组
}
target_df = pd.DataFrame(target_data)

knn(database_df, target_df)
