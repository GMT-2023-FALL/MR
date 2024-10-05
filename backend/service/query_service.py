import trimesh


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