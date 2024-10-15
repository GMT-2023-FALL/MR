from fastapi import HTTPException

from service.query_service import query_service


def query_controller(file, count):
    # 检查文件类型
    if not file.filename.endswith(".obj"):
        raise HTTPException(status_code=400, detail="only support .obj file.")
    try:
        return query_service(file, count)
    except Exception as e:
        print(e)
        # 返回错误信息
        raise HTTPException(status_code=500, detail=str(e))
