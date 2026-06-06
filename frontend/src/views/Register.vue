<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()
const form = reactive({ username: '', email: '', password: '', password2: '' })
const loading = ref(false)

async function handleRegister() {
  if (form.password !== form.password2) {
    return
  }
  loading.value = true
  try {
    await auth.register(form)
    router.push('/detect')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <el-card class="login-card">
      <template #header>
        <h2 style="text-align: center; margin: 0">Create Account</h2>
      </template>
      <el-form @submit.prevent="handleRegister">
        <el-form-item>
          <el-input v-model="form.username" placeholder="Username" size="large" prefix-icon="User" />
        </el-form-item>
        <el-form-item>
          <el-input v-model="form.email" placeholder="Email" size="large" prefix-icon="Message" />
        </el-form-item>
        <el-form-item>
          <el-input v-model="form.password" type="password" placeholder="Password" size="large" prefix-icon="Lock"
            show-password />
        </el-form-item>
        <el-form-item>
          <el-input v-model="form.password2" type="password" placeholder="Confirm password" size="large" prefix-icon="Lock"
            show-password @keyup.enter="handleRegister" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" size="large" :loading="loading" style="width: 100%"
            @click="handleRegister">Register</el-button>
        </el-form-item>
        <el-form-item style="text-align: center; margin-bottom: 0">
          <el-link type="primary" @click="router.push('/login')">Already have an account? Sign in</el-link>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<style scoped>
.login-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
}
.login-card {
  width: 420px;
  border-radius: 8px;
}
</style>
