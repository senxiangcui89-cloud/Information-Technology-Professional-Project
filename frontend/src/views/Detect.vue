<script setup lang="ts">
import { ref, onMounted } from "vue";
import { detectApi } from "../api";
import { ElMessage } from "element-plus";
import type { ModelInfo, DetectionBox } from "../types";

const models = ref<ModelInfo[]>([]);
const selectedModel = ref("yolo11n_raw");
const useClahe = ref(false);
const fileInput = ref<HTMLInputElement>();
const previewUrl = ref("");
const resultUrl = ref("");
const detections = ref<DetectionBox[]>([]);
const detectionCount = ref(0);
const inferenceTime = ref(0);
const loading = ref(false);
const hasResult = ref(false);
const taskId = ref(0);

const aiLoading = ref(false);
const aiAnalysis = ref("");
const reportLoading = ref(false);

onMounted(async () => {
  try {
    const res = await detectApi.getModels();
    models.value = res.data;
  } catch {
    /* handled by interceptor */
  }
});

function handleFileChange(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files?.[0]) {
    previewUrl.value = URL.createObjectURL(input.files[0]);
    hasResult.value = false;
    aiAnalysis.value = "";
  }
}

async function handleDetect() {
  if (!fileInput.value?.files?.[0]) {
    ElMessage.warning("Please select an image first");
    return;
  }
  loading.value = true;
  try {
    const form = new FormData();
    form.append("file", fileInput.value.files[0]);
    form.append("model_name", selectedModel.value);
    form.append("use_clahe", String(useClahe.value));

    const res = await detectApi.detectImage(form);
    detections.value = res.data.detections;
    detectionCount.value = res.data.detection_count;
    inferenceTime.value = res.data.inference_time_ms;
    resultUrl.value = res.data.result_image_url || "";
    taskId.value = res.data.task_id;
    hasResult.value = true;
    aiAnalysis.value = "";
    ElMessage.success(`Detection complete, found ${res.data.detection_count} targets`);
  } finally {
    loading.value = false;
  }
}

async function handleAnalyze() {
  if (!taskId.value) return;
  aiLoading.value = true;
  try {
    const res = await detectApi.analyze(taskId.value);
    aiAnalysis.value = res.data.analysis;
    ElMessage.success("AI analysis complete");
  } finally {
    aiLoading.value = false;
  }
}

async function handleExportReport() {
  if (!taskId.value) return;
  reportLoading.value = true;
  try {
    const res = await detectApi.exportReport(taskId.value);
    window.open(res.data.report_url, "_blank");
    ElMessage.success("Report generated");
  } finally {
    reportLoading.value = false;
  }
}
</script>

<template>
  <div>
    <h2>Image Debris Detection</h2>

    <el-card style="margin-bottom: 16px">
      <el-row :gutter="16" align="middle">
        <el-col :span="5">
          <el-select v-model="selectedModel" placeholder="Select model" style="width: 100%">
            <el-option
              v-for="m in models"
              :key="m.name"
              :label="`${m.display_name} (mAP50: ${m.mAP50})`"
              :value="m.name"
            />
          </el-select>
        </el-col>
        <el-col :span="3">
          <el-checkbox v-model="useClahe" border>CLAHE Enhance</el-checkbox>
        </el-col>
        <el-col :span="5">
          <input ref="fileInput" type="file" accept="image/*" @change="handleFileChange" style="font-size: 14px" />
        </el-col>
        <el-col :span="3">
          <el-button type="primary" :loading="loading" @click="handleDetect" style="width: 100%"> Detect </el-button>
        </el-col>
        <el-col :span="5">
          <el-tag v-if="hasResult" type="success" size="large">
            {{ detectionCount }} targets | {{ inferenceTime }}ms
          </el-tag>
        </el-col>
        <el-col :span="3">
          <template v-if="hasResult">
            <el-button
              text
              type="warning"
              :loading="aiLoading"
              @click="handleAnalyze"
              :disabled="detections.length === 0"
            >
              AI Analysis
            </el-button>
            <el-button text type="success" :loading="reportLoading" @click="handleExportReport">
              Export Report
            </el-button>
          </template>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card header="Original">
          <div v-if="previewUrl" style="text-align: center">
            <img :src="previewUrl" style="max-width: 100%; max-height: 500px; border-radius: 4px" />
          </div>
          <el-empty v-else description="Please select an image" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card header="Detection Result">
          <div v-if="hasResult && resultUrl" style="text-align: center">
            <img :src="resultUrl" style="max-width: 100%; max-height: 500px; border-radius: 4px" />
          </div>
          <el-empty v-else description="Awaiting detection" />
        </el-card>
      </el-col>
    </el-row>

    <el-card v-if="hasResult && detections.length > 0" style="margin-top: 16px" header="Detection Details">
      <el-table :data="detections" stripe max-height="400">
        <el-table-column prop="class_name" label="Class" width="100" />
        <el-table-column prop="confidence" label="Confidence" width="140">
          <template #default="{ row }">
            <el-progress
              :percentage="Number((row.confidence * 100).toFixed(1))"
              :stroke-width="16"
              :color="row.confidence > 0.7 ? '#67c23a' : row.confidence > 0.5 ? '#e6a23c' : '#f56c6c'"
            />
          </template>
        </el-table-column>
        <el-table-column label="Bounding Box" min-width="240">
          <template #default="{ row }"> ({{ row.x1 }}, {{ row.y1 }}) &rarr; ({{ row.x2 }}, {{ row.y2 }}) </template>
        </el-table-column>
        <el-table-column label="Size" width="100">
          <template #default="{ row }">
            {{ Math.round(row.x2 - row.x1) }} x {{ Math.round(row.y2 - row.y1) }} px
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="aiAnalysis" style="margin-top: 16px" header="AI Analysis">
      <div style="white-space: pre-wrap; line-height: 1.8; color: #333">{{ aiAnalysis }}</div>
    </el-card>
  </div>
</template>
