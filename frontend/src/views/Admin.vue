<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import http from '../api'

interface User {
  id: number; username: string; email: string; is_admin: boolean; created_at: string | null
}

const users = ref<User[]>([])
const loading = ref(false)

async function loadUsers() {
  loading.value = true
  try {
    const res = await http.get('/admin/users')
    users.value = res.data
  } finally { loading.value = false }
}

async function handleDelete(user: User) {
  try {
    await ElMessageBox.confirm(`Delete user "${user.username}"?`, 'Confirm Delete', { type: 'warning' })
    await http.delete(`/admin/users/${user.id}`)
    ElMessage.success('Deleted')
    loadUsers()
  } catch { /* cancelled */ }
}

async function handleToggleRole(user: User) {
  try {
    const action = user.is_admin ? 'Remove Admin' : 'Make Admin'
    await ElMessageBox.confirm(`${action} "${user.username}"?`, 'Confirm', { type: 'info' })
    await http.put(`/admin/users/${user.id}/role`)
    ElMessage.success('Role updated')
    loadUsers()
  } catch { /* cancelled */ }
}

onMounted(loadUsers)
</script>

<template>
  <div>
    <h2>User Management (Admin)</h2>

    <el-card>
      <el-table :data="users" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="username" label="Username" width="150" />
        <el-table-column prop="email" label="Email" min-width="200" />
        <el-table-column prop="is_admin" label="Role" width="100">
          <template #default="{ row }">
            <el-tag :type="row.is_admin ? 'danger' : 'info'">
              {{ row.is_admin ? 'Admin' : 'User' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="Registered" width="180">
          <template #default="{ row }">
            {{ row.created_at ? new Date(row.created_at).toLocaleString('en-US') : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="200">
          <template #default="{ row }">
            <el-button text type="warning" @click="handleToggleRole(row)">
              {{ row.is_admin ? 'Remove Admin' : 'Make Admin' }}
            </el-button>
            <el-button text type="danger" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>
