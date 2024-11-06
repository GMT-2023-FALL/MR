import time

import joblib
import numpy as np
import pandas as pd
from numba import njit
from pynndescent import NNDescent
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

NUM_VERTICES_THRESHOLD = 10000
# 加载归一化后的 CSV 文件
# df_normalized = pd.read_csv('./normalized_features.csv')
# df_normalized = pd.read_json('./normalized_features.json')
df_normalized = joblib.load('./normalized_features.pkl')
# 加载标准化器
scaler = joblib.load('./scaler_standard.pkl')
# 分离特征和标识列
feature_columns = [col for col in df_normalized.columns if col != 'Shape File']
X = df_normalized[feature_columns]
shape_files = df_normalized['Shape File'].values

# KNN prepared




# 加权值
# scalar_weights = {
#     'Surface Area': 0.025,
#     'Compactness': 0.025,
#     '3D Rectangularity': 0.1,
#     'Diameter': 0.1,
#     'Convexity': 0.1,
#     'Eccentricity': 0.15
# }
#
# vector_weights = {
#     'A3': 0.05,
#     'D1': 0.1,
#     'D2': 0.05,
#     'D3': 0.15,
#     'D4': 0.1
# }

scalar_weights = {
    'Surface Area': 0.05,
    'Compactness': 0.05,
    '3D Rectangularity': 0.1,
    'Diameter': 0.1,
    'Convexity': 0.1,
    'Eccentricity': 0.15
}

vector_weights = {
    'A3': 0.05,
    'D1': 0.2,
    'D2': 0.15,
    'D3': 0.1,
    'D4': 0.1
}

# 标量特征列
scalar_features = ['Surface Area', 'Compactness', '3D Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']

# 向量特征列
vector_features = ['A3', 'D1', 'D2', 'D3', 'D4']


def normalize_features(features):
    features[scalar_features] = scaler.fit_transform(features[scalar_features])
    # 2. 对向量特征列进行面积归一化（按元素数量）
    for col in vector_features:
        features[col] = features[col].apply(lambda x: [i / len(x) for i in x] if len(x) > 0 else x)

    return features


def extract_features(mesh):
    # 读取基本特质
    num_faces = len(mesh.faces)
    # check if the mesh needs to be refined
    if num_faces <= NUM_VERTICES_THRESHOLD:
        mesh = refine_mesh(mesh)


    # transform the mesh
    mesh = transform(mesh)


    # extract features
    features = extract_features_pipeline(mesh)

    features_df = pd.DataFrame([features])

    # TODO normalize the features
    normalized_features = normalize_features(features_df)


    return normalized_features


def refine_mesh(mesh, target_faces=10000):
    while len(mesh.faces) <= target_faces:
        mesh = mesh.subdivide()
    return mesh


def transform(mesh):
    _centroid = mesh.centroid
    mesh.vertices -= _centroid  # 平移到原点
    _max_extent = mesh.bounding_box.extents.max()  # 获取最大边长
    if _max_extent > 0:
        mesh.vertices /= _max_extent  # 统一缩放

    # Step 1: Compute PCA to get eigenvectors
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)

    # 获取前两个特征向量（主成分）
    eigenvectors = pca.components_

    # Step 2: Create a rotation matrix to align the first two eigenvectors with x and y axes
    # 计算旋转矩阵
    rotation_matrix = np.array([
        eigenvectors[0],  # 第一主成分对齐到 x 轴
        eigenvectors[1],  # 第二主成分对齐到 y 轴
        np.cross(eigenvectors[0], eigenvectors[1])  # 第三主成分对齐到 z 轴
    ])

    # Step 3: Apply the rotation to the vertices
    mesh.vertices = mesh.vertices @ rotation_matrix.T  # 应用旋转矩阵

    # Step 4: Flipping - Use moment test to determine if flipping is necessary
    for axis in range(3):
        if np.sum(mesh.vertices[:, axis] ** 3) < 0:
            mesh.vertices[:, axis] *= -1

    return mesh


def calculate_bins(n):
    # 使用平方根公式计算直方图的分箱数
    return int(np.sqrt(n))


num_samples = 100000
bins = calculate_bins(num_samples)


def extract_features_pipeline(mesh):
    features = {}

    # 基本描述符
    features['Surface Area'] = mesh.area
    features['Compactness'] = (36 * np.pi * (mesh.volume ** 2)) / (mesh.area ** 3) if mesh.area > 0 else 0
    obb = mesh.bounding_box_oriented
    features['3D Rectangularity'] = mesh.volume / obb.volume if obb.volume > 0 else 0
    features['Diameter'] = np.max(mesh.bounding_box.extents) if mesh.bounding_box.extents.size > 0 else 0
    convex_hull = mesh.convex_hull
    features['Convexity'] = mesh.volume / convex_hull.volume if convex_hull.volume > 0 else 0

    # 计算协方差矩阵和偏心率
    if mesh.vertices.shape[0] > 0:
        covariance_matrix = np.cov(mesh.vertices, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        features['Eccentricity'] = max(eigenvalues) / min(eigenvalues) if min(eigenvalues) > 0 else 0
    else:
        features['Eccentricity'] = 0

    # 随机选择的顶点
    random_indices = np.random.choice(mesh.vertices.shape[0], size=num_samples, replace=True)
    random_points = mesh.vertices[random_indices]

    # D1: 使用所有顶点与重心之间的距离
    barycenter = mesh.center_mass
    distances = np.linalg.norm(mesh.vertices - barycenter, axis=1)
    features['D1'] = np.histogram(distances[np.isfinite(distances)], bins=bins)[0]

    # A3: 随机选择3个顶点之间的角度
    v1 = random_points[np.random.choice(num_samples, num_samples, replace=True)]
    v2 = random_points[np.random.choice(num_samples, num_samples, replace=True)]
    v3 = random_points[np.random.choice(num_samples, num_samples, replace=True)]

    vec1 = v2 - v1
    vec2 = v3 - v1

    # 计算有效的长度
    norm_vec1 = np.linalg.norm(vec1, axis=1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)

    # 仅在长度有效时计算cos_theta
    valid_mask = (norm_vec1 > 0) & (norm_vec2 > 0)
    cos_theta = np.zeros(vec1.shape[0])
    cos_theta[valid_mask] = np.einsum('ij,ij->i', vec1[valid_mask], vec2[valid_mask]) / (
                norm_vec1[valid_mask] * norm_vec2[valid_mask])

    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    features['A3'] = np.histogram(angles[np.isfinite(angles)], bins=bins)[0]

    # D2: 两个随机顶点之间的距离
    v1 = random_points[np.random.choice(num_samples, num_samples, replace=True)]
    v2 = random_points[np.random.choice(num_samples, num_samples, replace=True)]
    d2_distances = np.linalg.norm(v1 - v2, axis=1)
    features['D2'] = np.histogram(d2_distances[np.isfinite(d2_distances)], bins=bins)[0]

    # D3: 随机选择3个顶点形成的三角形的面积的平方根
    areas = np.linalg.norm(np.cross(vec1, vec2), axis=1) / 2
    features['D3'] = np.histogram(np.sqrt(areas[np.isfinite(areas)]), bins=bins)[0]

    # D4: 由4个随机顶点形成的四面体的体积的立方根
    random_indices_4 = np.random.choice(mesh.vertices.shape[0], size=(num_samples, 4), replace=True)
    random_points_4 = mesh.vertices[random_indices_4]
    volumes = np.abs(np.einsum('ij,ik->i', random_points_4[:, 3] - random_points_4[:, 0],
                               np.cross(random_points_4[:, 1] - random_points_4[:, 0],
                                        random_points_4[:, 2] - random_points_4[:, 0]))) / 6
    volumes = volumes[volumes > 0]  # 确保点不共面
    features['D4'] = np.histogram(np.cbrt(volumes), bins=bins)[0]

    return features


def find_most_similar(query_features, method):
    """
    查找与查询特征最相似的形状。

    参数：
    - query_features: 查询形状的特征向量（未经归一化）
    - X: 数据库中所有形状的归一化特征向量
    - shape_files: 数据库中所有形状的文件名
    - scaler: 用于归一化查询特征的标准化器或最小-最大归一化器
    - method: 距离度量方法（'cosine' 或 'euclidean'）

    返回：
    - 最相似的形状文件名及其相似度
    """
    if method == 'default':
        return compute_distances_to_all(df_normalized,query_features)
    elif method == 'KNN':
       return knn(X, query_features)
    elif method == 'KDTree':
        return kd_tree(X, query_features)


def compute_weighted_distance(row_a, row_b):
    # 标量特征加权欧氏距离
    scalar_distance = 0
    for feature, weight in scalar_weights.items():
        dist = euclidean([row_a[feature]], [row_b[feature]])
        scalar_distance += weight * dist

    # 向量特征加权 EMD 距离
    vector_distance = 0
    for col, weight in vector_weights.items():
            hist1 = row_a[col]
            hist2 = row_b[col]
            emd_distance = wasserstein_distance(hist1, hist2)
            vector_distance += weight * emd_distance

    # 最终距离：标量距离与向量距离的加权和
    final_distance = scalar_distance + vector_distance
    return final_distance


def compute_distances_to_all(df, target_df):
    # external row_a should be the first row_a of the target_df with column names
    external_row = target_df.iloc[0]
    distances = []
    # only use the column that not equal to 'Shape File'

    for idx, df_row in df[feature_columns].iterrows():
        distance = compute_weighted_distance(external_row, df_row)
        # get the shape file by the idx
        file_path = df.iloc[idx]['Shape File'].split('/')[-1]
        genre = file_path.split('~')[0]
        name = file_path.split('~')[1]

        distances.append(
            {
                "file_path": "{}/{}".format(genre, name),
                "distance": distance
            }
        )
    # sort the distances by the distance
    distances = sorted(distances, key=lambda x: x['distance'])
    return distances


# 展平直方图特征，将每个数组展为多个列，同时保留其他非直方图列
def flatten_histograms(df):
    flattened_data = {}

    # 遍历 DataFrame 中的所有列
    for col in df.columns:
        if col in vector_weights.keys():  # 如果是直方图特征列
            # 使用 apply 方法展平每个直方图特征
            for i in range(len(df[col].iloc[0])):
                # 创建新的列名为 'column_index'，将直方图特征平铺
                flattened_data[f"{col}_{i}"] = df[col].apply(lambda x: x[i])
        else:  # 保留非直方图特征列
            flattened_data[col] = df[col]

    # 创建展平后的 DataFrame
    flattened_df = pd.DataFrame(flattened_data)

    # 返回展平后的 DataFrame
    return flattened_df

@njit
def custom_distance(a, b):
    return np.sum(np.abs(a - b))
    #return compute_weighted_distance_knn(a, b)


def compute_weighted_distance_knn(row_a, row_b):
    # 标量特征加权欧氏距离
    scalar_distance = 0
    for index, weight in enumerate(scalar_weights.values()):
        dist = euclidean([row_a[index]], [row_b[index]])
        scalar_distance += weight * dist

    # 向量特征加权 EMD 距离
    vector_distance = 0
    scalar_length = 6
    for index, weight in enumerate(vector_weights.values()):
        hist1 = row_a[scalar_length: scalar_length + (index + 1) * bins]
        hist2 = row_b[scalar_length: scalar_length + (index + 1) * bins]
        emd_distance = wasserstein_distance(hist1, hist2)
        vector_distance += weight * emd_distance
    # 最终距离：标量距离与向量距离的加权和
    final_distance = scalar_distance + vector_distance
    return final_distance


# 铺平
flatten_database_df = flatten_histograms(X)
# 计时
start_time = time.time()
# 使用 PyNNDescent 创建 ANN 查询引擎，替代 KD-Tree 的精确查询
# 设定 metric 为你的自定义距离，或选择适合的数据的默认距离（例如欧几里得距离）
nnd_index = NNDescent(flatten_database_df)
nnd_index.prepare()
print("ANN prepare time: {:.2f}s".format(time.time() - start_time))
t1 = time.time()
tree_index = KDTree(flatten_database_df)
# print the ms time
print("KDTree prepare time: {:.2f}ms".format((time.time() - t1) * 1000))



def knn(database_df, target_df):
    try :
        # external row_a should be the first row_a of the target_df with column names

        # 铺平
        # flatten_database_df = flatten_histograms(database_df)
        flatten_target_df = flatten_histograms(target_df)

        # 使用 PyNNDescent 创建 ANN 查询引擎，替代 KD-Tree 的精确查询
        # 设定 metric 为你的自定义距离，或选择适合的数据的默认距离（例如欧几里得距离）
        #index = NNDescent(flatten_database_df, metric='', n_neighbors=10)
        indices, distances = nnd_index.query(flatten_target_df)
        # 返回 shape file 中对应的文件名
        result = []

        for idx, distance in zip(indices.tolist()[0], distances.tolist()[0]):
            file_path = df_normalized.iloc[idx]['Shape File'].split('/')[-1]
            genre = file_path.split('~')[0]
            name = file_path.split('~')[1]
            result.append(
                {
                    "file_path": "{}/{}".format(genre, name),
                    "distance": distance
                }
            )
        return result
    except Exception as e:
        print(e)

def kd_tree(database_df, target_df):
    try :
        # external row_a should be the first row_a of the target_df with column names

        # 铺平
        # flatten_database_df = flatten_histograms(database_df)
        flatten_target_df = flatten_histograms(target_df)

        # 使用 PyNNDescent 创建 ANN 查询引擎，替代 KD-Tree 的精确查询
        # 设定 metric 为你的自定义距离，或选择适合的数据的默认距离（例如欧几里得距离）
        #index = NNDescent(flatten_database_df, metric='', n_neighbors=10)
        distances,indices = tree_index.query(flatten_target_df, k=10)
        # 返回 shape file 中对应的文件名
        result = []

        for idx, distance in zip(indices.tolist()[0], distances.tolist()[0]):
            file_path = df_normalized.iloc[idx]['Shape File'].split('/')[-1]
            genre = file_path.split('~')[0]
            name = file_path.split('~')[1]
            result.append(
                {
                    "file_path": "{}/{}".format(genre, name),
                    "distance": distance
                }
            )
        print(result)
        return result
    except Exception as e:
        print(e)


# # print(knn(X, X.iloc[0:1]))
# t2 = time.time()
# compute_distances_to_all(df_normalized, X.iloc[0:1])
# print("default query time: {:.2f}ms".format((time.time() - t2) * 1000))