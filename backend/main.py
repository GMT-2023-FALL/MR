import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from controller.query_controller import query_controller

app = FastAPI()

# 配置允许的来源
origins = [
    # 允许所有
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源
    allow_credentials=True,
    allow_methods=["*"],     # 允许的HTTP方法
    allow_headers=["*"],     # 允许的HTTP头
)

@app.post("/api/query")
#@app.post("/query")
async def upload_obj(file: UploadFile = File(...),count: int = Form(5)):
    return query_controller(file, count)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
