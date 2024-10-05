import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置允许的来源
origins = [
    # 允许所有
    "*"
    # 你可以根据需要添加更多的来源
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源
    allow_credentials=True,
    allow_methods=["*"],     # 允许的HTTP方法
    allow_headers=["*"],     # 允许的HTTP头
)

@app.post("/api/query")
async def upload_obj(
    file: UploadFile = File(...),
    count: int = Form(3),
    t: float = Form(1.0)):
    if not file.filename.endswith(".obj"):
        raise HTTPException(status_code=400, detail="只支持 .obj 文件。")
    try:
        # TODO 读取.obj文件，并处理
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
        return mock_response
    except Exception as e:
        # 返回错误信息
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
