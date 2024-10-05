# 使用官方 Node.js 18 轻量级镜像作为基础镜像
FROM node:18-alpine

# 设置工作目录
WORKDIR /app

# 复制前端依赖文件并安装
COPY package.json ./
RUN npm install

# 复制前端源代码
COPY . .

# 启动 Vite 开发服务器
CMD ["npm", "run", "dev"]
