<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import http from '../api'

interface ModelOpt { name: string; display_name: string; mAP50: number }
interface DatasetOpt { name: string; train_images: number; val_images: number }
interface EvalTask {
  task_id: number; status: string; model_name: string; dataset_name: string
  image_count: number; metrics: any; created_at: string | null
}

const models = ref<ModelOpt[]>([])
const datasets = ref<DatasetOpt[]>([])
const selectedModel = ref('yolo11n_raw')
const selectedDataset = ref('')
const evaling = ref(false)
const currentTask = ref<EvalTask | null>(null)
const tasks = ref<EvalTask[]>([])
const loading = ref(false)
const chartRef = ref<HTMLDivElement>()

async function loadOptions() {
  const res = await http.get('/eval/options')
  models.value = res.data.models
  datasets.value = res.data.datasets
  if (datasets.value.length > 0) selectedDataset.value = datasets.value[0].name
}
async function loadTasks() {
  loading.value = true
  try {
    const res = await http.get('/eval/tasks')
    tasks.value = res.data
  } finally { loading.value = false }
}

onMounted(() => { loadOptions(); loadTasks() })

async function handleEval() {
  if (!selectedDataset.value) {
    ElMessage.warning('Please upload a dataset first')
    return
  }
  evaling.value = true
  try {
    const form = new FormData()
    form.append('model_name', selectedModel.value)
    form.append('dataset_name', selectedDataset.value)
    const res = await http.post('/eval/run', form)
    ElMessage.success('Evaluation task started')

    // Poll until done
    const poll = setInterval(async () => {
      const r = await http.get(`/eval/result/${res.data.task_id}`)
      if (r.data.status === 'done' || r.data.status === 'failed') {
        clearInterval(poll)
        currentTask.value = r.data
        evaling.value = false
        loadTasks()
        if (r.data.status === 'done') {
          ElMessage.success('Evaluation complete')
          // Render chart after DOM update
          setTimeout(renderChart, 200)
        } else {
          ElMessage.error('Evaluation failed: ' + (r.data.error_message || 'Unknown'))
        }
      }
    }, 2000)
  } catch { evaling.value = false }
}

function renderChart() {
  if (!chartRef.value || !currentTask.value?.metrics) return
  const chart = echarts.init(chartRef.value)
  const m = currentTask.value.metrics
  chart.setOption({
    title: { text: `${currentTask.value.model_name} @ ${currentTask.value.dataset_name}` },
    tooltip: { trigger: 'axis' },
    xAxis: { data: ['mAP50', 'mAP50-95', 'Precision', 'Recall'] },
    yAxis: { max: 1, axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' } },
    series: [{
      type: 'bar', barWidth: 80,
      data: [
        { value: m.mAP50, itemStyle: { color: '#409eff' } },
        { value: m.mAP50_95, itemStyle: { color: '#67c23a' } },
        { value: m.precision, itemStyle: { color: '#e6a23c' } },
        { value: m.recall, itemStyle: { color: '#f56c6c' } },
      ],
      label: { show: true, formatter: (p: any) => (p.value * 100).toFixed(2) + '%' },
    }],
    grid: { top: 60, bottom: 30 },
  })
  window.addEventListener('resize', () => chart.resize())
}

async function handleViewResult(task: EvalTask) {
  currentTask.value = task
  if (task.metrics) setTimeout(renderChart, 200)
}
</script>

<template>
  <div>
    <h2>Model Evaluation</h2>

    <el-card style="margin-bottom: 16px">
      <el-row :gutter="16" align="middle">
        <el-col :span="6">
          <el-select v-model="selectedModel" placeholder="Select model" style="width: 100%">
            <el-option v-for="m in models" :key="m.name" :label="m.display_name" :value="m.name" />
          </el-select>
        </el-col>
        <el-col :span="6">
          <el-select v-model="selectedDataset" placeholder="Select dataset" style="width: 100%">
            <el-option v-for="d in datasets" :key="d.name" :label="`${d.name} (${d.val_images} val)`" :value="d.name" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-button type="primary" :loading="evaling" :disabled="datasets.length === 0"
            @click="handleEval" style="width: 100%">
            Start Evaluation
          </el-button>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="14">
        <el-card v-if="currentTask?.metrics" header="Evaluation Metrics">
          <div ref="chartRef" style="height: 350px"></div>
          <el-descriptions :column="4" border style="margin-top: 16px">
            <el-descriptions-item label="mAP50">{{ currentTask.metrics.mAP50 }}</el-descriptions-item>
            <el-descriptions-item label="mAP50-95">{{ currentTask.metrics.mAP50_95 }}</el-descriptions-item>
            <el-descriptions-item label="Precision">{{ currentTask.metrics.precision }}</el-descriptions-item>
            <el-descriptions-item label="Recall">{{ currentTask.metrics.recall }}</el-descriptions-item>
            <el-descriptions-item label="FPS">{{ currentTask.metrics.fps || 'N/A' }}</el-descriptions-item>
            <el-descriptions-item label="Inference Time">{{ currentTask.metrics.inference_ms }}ms</el-descriptions-item>
            <el-descriptions-item label="Model">{{ currentTask.model_name }}</el-descriptions-item>
            <el-descriptions-item label="Dataset">{{ currentTask.dataset_name }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
        <el-empty v-else description="Select a model and dataset, then click Start Evaluation" />
      </el-col>
      <el-col :span="10">
        <el-card header="Evaluation History" style="max-height: 600px; overflow-y: auto">
          <el-table :data="tasks" stripe v-loading="loading" empty-text="No evaluation records" max-height="500">
            <el-table-column prop="task_id" label="ID" width="60" />
            <el-table-column prop="model_name" label="Model" min-width="140" />
            <el-table-column prop="dataset_name" label="Dataset" min-width="120" />
            <el-table-column prop="status" label="Status" width="90">
              <template #default="{ row }">
                <el-tag :type="row.status === 'done' ? 'success' : row.status === 'failed' ? 'danger' : 'warning'">
                  {{ row.status === 'done' ? 'Done' : row.status === 'failed' ? 'Failed' : 'Running' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="60">
              <template #default="{ row }">
                <el-button text type="primary" @click="handleViewResult(row)"
                  :disabled="row.status !== 'done'">View</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>
