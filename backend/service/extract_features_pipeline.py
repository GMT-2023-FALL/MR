import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

NUM_FACES_THRESHOLD = 100
# 加载归一化后的 CSV 文件
df_normalized = pd.read_csv('./normalized_features.csv')
# 加载标准化器
scaler = joblib.load('./scaler_standard.pkl')
# 分离特征和标识列
feature_columns = [col for col in df_normalized.columns if col.startswith('scalar_') or col.startswith('vector_')]
X = df_normalized[feature_columns].values
shape_files = df_normalized['Shape File'].values


def extract_features(mesh):
    # 读取基本特质
    num_faces = len(mesh.faces)
    # check if the mesh needs to be refined
    if num_faces <= NUM_FACES_THRESHOLD:
        mesh = refine_mesh(mesh)
    # normalized the mesh
    mesh = normalize_mesh(mesh)
    # extract features
    features = extract_features_pipeline(mesh)

    return features


def refine_mesh(mesh, target_faces=500):
    while len(mesh.faces) < target_faces:
        mesh = mesh.subdivide()
    return mesh


def normalize_mesh(mesh):
    #标准化模型
    centroid = mesh.centroid

    mesh.vertices -= centroid  # 平移到原点

    max_extent = mesh.bounding_box.extents.max()  # 获取最大边长
    if max_extent > 0:
        mesh.vertices /= max_extent

    # Centering (already implemented in Step 2)
    mesh.vertices -= mesh.center_mass

    # Scaling (already implemented in Step 2)
    scale = np.cbrt(1.0 / mesh.volume)
    mesh.vertices *= scale

    # Alignment using PCA
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    aligned_vertices = pca.transform(mesh.vertices)
    mesh.vertices = aligned_vertices

    # Flipping using Moment Test
    for axis in range(3):
        if np.sum(mesh.vertices[:, axis]) < 0:
            mesh.vertices[:, axis] *= -1

    return mesh


def extract_features_pipeline(mesh):
        features = {}
        # Elementary Descriptors
        features['Surface Area'] = mesh.area
        features['Compactness'] = (36 * np.pi * (mesh.volume ** 2)) / (
                    mesh.area ** 3)  # Compactness with respect to a sphere
        obb = mesh.bounding_box_oriented
        features['3D Rectangularity'] = mesh.volume / obb.volume
        features['Diameter'] = np.max(mesh.bounding_box.extents)
        convex_hull = mesh.convex_hull
        features['Convexity'] = mesh.volume / convex_hull.volume
        covariance_matrix = np.cov(mesh.vertices, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        features['Eccentricity'] = max(eigenvalues) / min(eigenvalues)

        # Property Descriptors (using histograms)
        num_samples = 1000
        bins = 10  # Number of bins for the histogram

        random_points = mesh.vertices[np.random.choice(mesh.vertices.shape[0], num_samples, replace=True)]

        # A3: Angle between 3 random vertices
        angles = []
        for i in range(num_samples):
            v1, v2, v3 = random_points[np.random.choice(num_samples, 3, replace=False)]
            vec1 = v2 - v1
            vec2 = v3 - v1
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if -1.0 <= cos_theta <= 1.0:  # Ensure valid range for arccos
                angles.append(np.arccos(cos_theta))
        features['A3'] = np.histogram(angles, bins=bins, range=(0, np.pi))[0]

        # D1: Distance between barycenter and random vertex
        barycenter = mesh.center_mass
        distances = np.linalg.norm(random_points - barycenter, axis=1)
        features['D1'] = np.histogram(distances, bins=bins)[0]

        # D2: Distance between 2 random vertices
        d2_distances = []
        for i in range(num_samples // 2):
            v1, v2 = random_points[np.random.choice(num_samples, 2, replace=False)]
            d2_distances.append(np.linalg.norm(v1 - v2))
        features['D2'] = np.histogram(d2_distances, bins=bins)[0]

        # D3: Square root of area of triangle given by 3 random vertices
        areas = []
        for i in range(num_samples // 3):
            v1, v2, v3 = random_points[np.random.choice(num_samples, 3, replace=False)]
            if not np.allclose(np.cross(v2 - v1, v3 - v1), 0):  # Ensure points are not collinear
                area = np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2
                areas.append(np.sqrt(area))
        features['D3'] = np.histogram(areas, bins=bins)[0]

        # D4: Cube root of volume of tetrahedron formed by 4 random vertices
        volumes = []
        for i in range(num_samples // 4):
            retries = 0
            while retries < 5:  # Try up to 5 times to find non-coplanar points
                v1, v2, v3, v4 = random_points[np.random.choice(num_samples, 4, replace=False)]
                volume = np.abs(np.dot(v4 - v1, np.cross(v2 - v1, v3 - v1))) / 6
                if volume > 0:  # Ensure points are not coplanar
                    volumes.append(np.cbrt(volume))
                    break
                retries += 1
        features['D4'] = np.histogram(volumes, bins=bins)[0]

        # 标量特征列
        scalar_features = ['Surface Area', 'Compactness', '3D Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']

        # 向量特征列
        vector_features = ['A3', 'D1', 'D2', 'D3', 'D4']

        # 创建特征向量
        query_scalar = np.array([
            features['Surface Area'],
            features['Compactness'],
            features['3D Rectangularity'],
            features['Diameter'],
            features['Convexity'],
            features['Eccentricity']
        ])

        # 处理向量特征
        query_vector = []
        for col in vector_features:
            vec = np.array(features[col], dtype=float)  # 确保转换为浮点数
            if len(vec) < 10:
                vec = np.pad(vec, (0, 10 - len(vec)), 'constant')
            else:
                vec = vec[:10]
            query_vector.extend(vec)
        query_vector = np.array(query_vector)

        # 合并所有特征
        query_combined = np.hstack([query_scalar, query_vector])

        # 标准化标量特征（使用之前的 scaler_standard）
        query_combined[:6] = scaler.transform(query_scalar.reshape(1, -1))

        return query_combined



def find_most_similar(query_features, x=X, _shape_files=shape_files, _scaler=scaler, method='euclidean'):
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

    # 预处理查询特征
    scalar_features = query_features[:6].reshape(1, -1)
    vector_features = query_features[6:]

    # 标准化标量特征（假设使用标准化）
    if scaler:
        scalar_features = scaler.transform(scalar_features)

    # 合并特征
    query_combined = np.hstack([scalar_features, vector_features.reshape(1, -1)])

    # 计算相似度
    if method == 'cosine':
        similarities = cosine_similarity(query_combined, x)
    elif method == 'euclidean':
        from sklearn.metrics import pairwise_distances
        similarities = -pairwise_distances(query_combined, x, metric='euclidean')  # 负距离作为相似度
    else:
        raise ValueError("Unsupported method. Choose 'cosine' or 'euclidean'.")


    # find 10 most similar shapes
    most_similar_indices = np.argsort(similarities[0])[::-1][:10]
    most_similar_shapes = _shape_files[most_similar_indices]
    similarity_scores = similarities[0][most_similar_indices]

    result = [ {"file_path": shape_file, "distance": distance} for shape_file, distance in zip(most_similar_shapes, similarity_scores) ]

    return result




