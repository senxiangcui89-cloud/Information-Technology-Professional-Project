import { defineStore } from 'pinia'
import { ref } from 'vue'
import { authApi } from '../api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const username = ref(localStorage.getItem('username') || '')
  const isAdmin = ref(localStorage.getItem('isAdmin') === 'true')

  async function login(form: { username: string; password: string }) {
    const res = await authApi.login(form)
    token.value = res.data.access_token
    username.value = res.data.username
    isAdmin.value = res.data.is_admin
    localStorage.setItem('token', res.data.access_token)
    localStorage.setItem('username', res.data.username)
    localStorage.setItem('isAdmin', String(res.data.is_admin))
  }

  async function register(form: { username: string; email: string; password: string }) {
    const res = await authApi.register(form)
    token.value = res.data.access_token
    username.value = res.data.username
    isAdmin.value = res.data.is_admin
    localStorage.setItem('token', res.data.access_token)
    localStorage.setItem('username', res.data.username)
    localStorage.setItem('isAdmin', String(res.data.is_admin))
  }

  function logout() {
    token.value = ''
    username.value = ''
    isAdmin.value = false
    localStorage.clear()
  }

  return { token, username, isAdmin, login, register, logout }
})
