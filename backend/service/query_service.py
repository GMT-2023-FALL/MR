import trimesh

from service.extract_features_pipeline import extract_features, find_most_similar


def query_service(file, count, method):
    try:

        #读取.obj文件，并处理
        file.file.seek(0)
        # 使用 trimesh 加载上传的 .obj 文件
        mesh = trimesh.load(file.file, file_type='obj')
        target_features = extract_features(mesh)
        result = find_most_similar(target_features, method)
        # 返回查询后的数据
        if count > len(result):
            count = len(result)

        return result[:count]
    except Exception as e:
        print(e)


    # # 模拟返回一个固定的 JSON 列表
    # mock_response = [
    #     {"file_path": './normalized_database/Bicycle/D00040.obj', "distance": 1.0},
    #     {"file_path": './normalized_database/Bird/D00442.obj', "distance": 2.0},
    #     {"file_path": './normalized_database/AircraftBuoyant/m1341.obj', "distance": 3.0},
    #     {"file_path": './normalized_database/Apartment/D00045.obj', "distance": 4.0},
    #     {"file_path": './normalized_database/AquaticAnimal/m78.obj', "distance": 5.0},
    #     {"file_path": './normalized_database/Biplane/D00276.obj', "distance": 6.0},
    # ]
    # # 返回前 count 个结果
    # if count > len(mock_response):
    #     count = len(mock_response)
    # return mock_response[:count]
