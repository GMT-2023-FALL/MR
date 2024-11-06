# 使用官方 Python 3.8 镜像作为基础镜像
FROM python:3.8

# 设置工作目录
WORKDIR /app

# 复制后端依赖文件并安装
COPY requirements.txt ./
RUN pip install -U pip
RUN pip install -r requirements.txt

# 复制后端源代码
COPY . .

# 暴露端口（假设 FastAPI 运行在 8000 端口）
EXPOSE 8000

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
