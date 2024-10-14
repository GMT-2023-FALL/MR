

NUM_FACES_THRESHOLD = 100


def preload_dataset():


    #TODO 读取数据集

    return



def extract_features(mesh):
    #TODO 读取基本特质
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    # check if the mesh needs to be refined
    if(num_faces <= NUM_FACES_THRESHOLD):
        mesh = refine_mesh(mesh)
    # normalized the mesh
    mesh = normalize_mesh(mesh)
    # extract features
    features = extract_features_pipeline(mesh)
    return features

def refine_mesh(mesh):
    #TODO 优化模型

    return


def normalize_mesh(mesh):
    #TODO 标准化模型

    return


def extract_features_pipeline(mesh):

    return

