import trimesh
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 预读取
# df = pd.read_csv('normalized_database.csv')
# scaler = StandardScaler()
# # 选择特征
# feature_columns = ['num_vertices', 'num_faces', 'face_types', 'bounding_box']
# X = df[feature_columns]
# # 标准化
# X_scaled = scaler.fit_transform(X)


# def get_distance(target_features, method='euclidean'):
#     # 选择计算方法
#     if method == 'euclidean':
#         distances = pairwise_distances([target_features], X_scaled, metric='euclidean')[0]
#     elif method == 'cosine':
#         distances = pairwise_distances([target_features], X_scaled, metric='cosine')[0]
#     elif method == 'earth_mover':
#         distances = pairwise_distances([target_features], X_scaled, metric='manhattan')[0]
#     else:
#         raise ValueError("Invalid method")
#     distance_df = pd.DataFrame({
#         'index': df.index,
#         'distance': distances
#     })
#     # 获取距离最小的 10 个条目
#     top_10 = distance_df.nsmallest(10, 'distance').copy()
#     # 获取top_10 的文件路径
#     top_10['file_path'] = df.loc[top_10['index'], 'file_path'].values
#     # 将文件路径和距离转换为字典
#     result = top_10[['file_path', 'distance']].to_dict(orient='records')
#     return result


def query_service(file, count):
    # TODO 读取.obj文件，并处理
    file.file.seek(0)
    # 使用 trimesh 加载上传的 .obj 文件
    mesh = trimesh.load(file.file, file_type='obj')
    # num_vertices = len(mesh.vertices)
    # num_faces = len(mesh.faces)
    # face_types = mesh.faces.shape[1]  # 3 for triangles, 4 for quads
    # bounding_box = mesh.bounding_box.extents.tolist()
    # print("Number of vertices: ", num_vertices, "Number of faces: ", num_faces, "Face types: ", face_types,
    #       "Bounding box: ", bounding_box)
    # TODO 返回查询后的数据
    # 模拟返回一个固定的 JSON 列表
    mock_response = [
        {"file_path": './normalized_database/Bicycle/D00040.obj', "distance": 1.0},
        {"file_path": './normalized_database/Bird/D00442.obj', "distance": 2.0},
        {"file_path": './normalized_database/AircraftBuoyant/m1341.obj', "distance": 3.0},
        {"file_path": './normalized_database/Apartment/D00045.obj', "distance": 4.0},
        {"file_path": './normalized_database/AquaticAnimal/m78.obj', "distance": 5.0},
        {"file_path": './normalized_database/Biplane/D00276.obj', "distance": 6.0},
    ]
    # 返回前 count 个结果
    if count > len(mock_response):
        count = len(mock_response)
    return mock_response[:count]
