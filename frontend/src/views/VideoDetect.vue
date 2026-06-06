<script setup lang="ts">
import { ref, onUnmounted } from "vue";
import { detectApi } from "../api";
import { ElMessage } from "element-plus";
import type { ModelInfo } from "../types";

const models = ref<ModelInfo[]>([]);
const selectedModel = ref("yolo11n_raw");
const useClahe = ref(false);
const fileInput = ref<HTMLInputElement>();
const loading = ref(false);
const processing = ref(false);
const taskId = ref<number | null>(null);
const resultVideoUrl = ref("");
const progress = ref({ current: 0, total: 0, status: "" });
const avgInference = ref(0);
const totalDetections = ref(0);
let pollTimer: ReturnType<typeof setInterval> | null = null;

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});

async function loadModels() {
  try {
    const res = await detectApi.getModels();
    models.value = res.data;
  } catch {
    /* */
  }
}
loadModels();

async function handleUpload() {
  if (!fileInput.value?.files?.[0]) {
    ElMessage.warning("Please select a video file first");
    return;
  }
  loading.value = true;
  try {
    const form = new FormData();
    form.append("file", fileInput.value.files[0]);
    form.append("model_name", selectedModel.value);
    form.append("use_clahe", String(useClahe.value));

    const res = await detectApi.detectVideo(form);
    taskId.value = res.data.task_id;
    processing.value = true;
    ElMessage.info(res.data.message || "Video processing started");

    // Poll progress
    pollTimer = setInterval(async () => {
      try {
        const r = await detectApi.getProgress(taskId.value!);
        progress.value = r.data;
        if (r.data.status === "done") {
          clearInterval(pollTimer!);
          pollTimer = null;
          processing.value = false;
          avgInference.value = r.data.avg_inference_ms || 0;
          totalDetections.value = r.data.total_detections || 0;
          const tr = await detectApi.getResult(taskId.value!);
          resultVideoUrl.value = tr.data.result_video_url || "";
          ElMessage.success(`Processing complete! Detected ${totalDetections.value} targets total`);
        }
      } catch {
        clearInterval(pollTimer!);
        pollTimer = null;
        processing.value = false;
        const tr = await detectApi.getResult(taskId.value!);
        if (tr.data.status === "done") {
          resultVideoUrl.value = tr.data.result_video_url || "";
        }
      }
    }, 1500);
  } finally {
    loading.value = false;
  }
}

function progressPercent() {
  if (progress.value.total === 0) return 0;
  return Math.round((progress.value.current / progress.value.total) * 100);
}
</script>

<template>
  <div>
    <h2>Video Debris Detection</h2>

    <el-card style="margin-bottom: 16px">
      <el-row :gutter="16" align="middle">
        <el-col :span="6">
          <el-select v-model="selectedModel" placeholder="Select model" style="width: 100%">
            <el-option
              v-for="m in models"
              :key="m.name"
              :label="`${m.display_name} (mAP50: ${m.mAP50})`"
              :value="m.name"
            />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-checkbox v-model="useClahe" border>CLAHE Enhance</el-checkbox>
        </el-col>
        <el-col :span="6">
          <input ref="fileInput" type="file" accept="video/*" style="font-size: 14px" />
        </el-col>
        <el-col :span="4">
          <el-button type="primary" :loading="loading" :disabled="processing" @click="handleUpload" style="width: 100%">
            {{ processing ? "Processing..." : "Detect" }}
          </el-button>
        </el-col>
      </el-row>
    </el-card>

    <el-card v-if="processing" style="margin-bottom: 16px">
      <template #header>Progress</template>
      <div style="text-align: center">
        <el-progress
          :percentage="progressPercent()"
          :stroke-width="20"
          :status="progressPercent() === 100 ? 'success' : undefined"
        />
        <p style="margin-top: 12px; color: #606266">
          Frame {{ progress.current }} / {{ progress.total }}
          <span v-if="progress.avg_inference_ms"> | Avg inference: {{ progress.avg_inference_ms }}ms/frame</span>
        </p>
      </div>
    </el-card>

    <el-card v-if="resultVideoUrl">
      <template #header>
        <span>Detection Result</span>
        <el-tag type="success" style="margin-left: 12px">
          {{ totalDetections }} targets | {{ avgInference }}ms/frame
        </el-tag>
      </template>
      <div style="text-align: center">
        <video :src="resultVideoUrl" controls style="max-width: 100%; max-height: 500px; border-radius: 4px" />
      </div>
    </el-card>

    <el-empty v-if="!processing && !resultVideoUrl" description="Select a video file and click Detect" />
  </div>
</template>
