import axios from 'axios'
import { ElMessage } from 'element-plus'

const http = axios.create({ baseURL: '/api' })

http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

http.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || 'Request failed'
    ElMessage.error(msg)
    if (err.response?.status === 401) {
      localStorage.clear()
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export const authApi = {
  login: (data: { username: string; password: string }) => http.post('/auth/login', data),
  register: (data: { username: string; email: string; password: string }) => http.post('/auth/register', data),
  me: () => http.get('/auth/me'),
}

export const detectApi = {
  getModels: () => http.get('/detect/models'),
  detectImage: (form: FormData) => http.post('/detect/image', form),
  detectVideo: (form: FormData) => http.post('/detect/video', form),
  getResult: (taskId: number) => http.get(`/detect/result/${taskId}`),
  getTasks: () => http.get('/detect/tasks'),
  getProgress: (taskId: number) => http.get(`/detect/progress/${taskId}`),
  analyze: (taskId: number, apiKey?: string) => {
    const form = new FormData()
    form.append('task_id', String(taskId))
    if (apiKey) form.append('api_key', apiKey)
    return http.post('/detect/analyze', form)
  },
  exportReport: (taskId: number) => http.get(`/detect/export-report/${taskId}`),
  detectFrame: (form: FormData) => http.post('/camera/frame', form),
}

export default http
