<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import http from '../api'

interface DatasetInfo {
  name: string
  valid: boolean
  train_images: number
  val_images: number
  train_labels: number
  val_labels: number
  structure_ok: boolean
}

const datasets = ref<DatasetInfo[]>([])
const loading = ref(false)
const uploading = ref(false)
const fileInput = ref<HTMLInputElement>()

async function loadDatasets() {
  loading.value = true
  try {
    const res = await http.get('/dataset/list')
    datasets.value = res.data
  } finally {
    loading.value = false
  }
}

async function handleUpload() {
  if (!fileInput.value?.files?.[0]) {
    ElMessage.warning('Please select a ZIP file')
    return
  }
  uploading.value = true
  try {
    const form = new FormData()
    form.append('file', fileInput.value.files[0])
    await http.post('/dataset/upload', form)
    ElMessage.success('Dataset uploaded successfully')
    fileInput.value.value = ''
    loadDatasets()
  } finally {
    uploading.value = false
  }
}

async function handleDelete(name: string) {
  try {
    await ElMessageBox.confirm(`Delete dataset "${name}"?`, 'Confirm', { type: 'warning' })
    await http.delete(`/dataset/${name}`)
    ElMessage.success('Deleted')
    loadDatasets()
  } catch { /* cancelled */ }
}

onMounted(loadDatasets)
</script>

<template>
  <div>
    <h2>Dataset Manager</h2>

    <el-card style="margin-bottom: 16px">
      <el-row :gutter="16" align="middle">
        <el-col :span="12">
          <input ref="fileInput" type="file" accept=".zip" style="font-size: 14px" />
          <span style="margin-left: 8px; color: #909399; font-size: 13px">
            ZIP format, must contain images/train, images/val, labels/train, labels/val
          </span>
        </el-col>
        <el-col :span="4">
          <el-button type="primary" :loading="uploading" @click="handleUpload">Upload Dataset</el-button>
        </el-col>
      </el-row>
    </el-card>

    <el-card>
      <template #header>Dataset List ({{ datasets.length }})</template>
      <el-table :data="datasets" stripe v-loading="loading" empty-text="No datasets available. Upload a ZIP file">
        <el-table-column prop="name" label="Name" min-width="200" />
        <el-table-column prop="structure_ok" label="Structure" width="100">
          <template #default="{ row }">
            <el-tag :type="row.structure_ok ? 'success' : 'danger'">
              {{ row.structure_ok ? 'Valid' : 'Invalid' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="train_images" label="Train Images" width="110" />
        <el-table-column prop="train_labels" label="Train Labels" width="110" />
        <el-table-column prop="val_images" label="Val Images" width="110" />
        <el-table-column prop="val_labels" label="Val Labels" width="110" />
        <el-table-column label="Actions" width="100">
          <template #default="{ row }">
            <el-button type="danger" text @click="handleDelete(row.name)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>
