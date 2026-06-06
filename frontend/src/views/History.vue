<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { detectApi } from '../api'

interface TaskRecord {
  task_id: number
  model_name: string
  source_type: string
  source_filename: string
  status: string
  detection_count: number
  inference_time_ms: number
  created_at: string
}

const tasks = ref<TaskRecord[]>([])
const loading = ref(false)

onMounted(async () => {
  loading.value = true
  try {
    const res = await detectApi.getTasks()
    tasks.value = res.data
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div>
    <h2>Detection History</h2>
    <el-card>
      <el-table :data="tasks" stripe v-loading="loading" empty-text="No records">
        <el-table-column prop="task_id" label="Task ID" width="80" />
        <el-table-column prop="model_name" label="Model" width="180" />
        <el-table-column prop="source_filename" label="Source File" min-width="200" />
        <el-table-column prop="source_type" label="Type" width="80" />
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'done' ? 'success' : row.status === 'failed' ? 'danger' : 'warning'">
              {{ row.status === 'done' ? 'Done' : row.status === 'failed' ? 'Failed' : row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="detection_count" label="Detections" width="80" />
        <el-table-column prop="inference_time_ms" label="Time(ms)" width="100" />
        <el-table-column prop="created_at" label="Time" width="180">
          <template #default="{ row }">
            {{ row.created_at ? new Date(row.created_at).toLocaleString('en-US') : '-' }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>
