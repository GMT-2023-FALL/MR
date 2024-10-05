import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000, // 这里指定你想要的端口
    proxy: {
      '/api': {
        target: 'http://backend:8000',  // Docker Compose 内部网络中的后端服务
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
