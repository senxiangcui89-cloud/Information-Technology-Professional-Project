<script setup lang="ts">
import { ref, onUnmounted, nextTick } from "vue";
import { detectApi } from "../api";
import { ElMessage } from "element-plus";
import type { ModelInfo } from "../types";

const models = ref<ModelInfo[]>([]);
const selectedModel = ref("yolo11n_raw");
const streaming = ref(false);
const videoRef = ref<HTMLVideoElement>();
const canvasRef = ref<HTMLCanvasElement>();
const detections = ref<{ class_name: string; confidence: number }[]>([]);
const count = ref(0);
const fps = ref(0);
const inferenceMs = ref(0);
let stream: MediaStream | null = null;
let intervalId: ReturnType<typeof setInterval> | null = null;
let lastTime = 0;

async function loadModels() {
  try {
    const r = await detectApi.getModels();
    models.value = r.data;
  } catch {
    /* */
  }
}
loadModels();

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "environment" },
    });
    if (videoRef.value) {
      videoRef.value.srcObject = stream;
      await videoRef.value.play();
    }
    streaming.value = true;
    ElMessage.success("Camera started");

    // Start sending frames
    lastTime = performance.now();
    intervalId = setInterval(captureAndSend, 300); // ~3 FPS sending
  } catch (err: any) {
    ElMessage.error("Camera failed: " + (err.message || "Permission denied"));
  }
}

function stopCamera() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  streaming.value = false;
  count.value = 0;
  detections.value = [];
}

async function captureAndSend() {
  if (!videoRef.value || !canvasRef.value) return;
  const canvas = canvasRef.value;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = videoRef.value.videoWidth;
  canvas.height = videoRef.value.videoHeight;
  ctx.drawImage(videoRef.value, 0, 0);

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
  if (!blob) return;

  const reader = new FileReader();
  const b64 = await new Promise<string>((resolve) => {
    reader.onload = () => resolve(reader.result as string);
    reader.readAsDataURL(blob);
  });

  try {
    const form = new FormData();
    form.append("image_b64", b64.split(",")[1]);
    form.append("model_name", selectedModel.value);
    const res = await detectApi.detectFrame(form);

    count.value = res.data.count;
    detections.value = res.data.detections;
    inferenceMs.value = res.data.inference_ms;

    // Draw result on canvas
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
    };
    img.src = "data:image/jpeg;base64," + res.data.image_b64;

    const now = performance.now();
    fps.value = Math.round(1000 / (now - lastTime));
    lastTime = now;
  } catch {
    /* skip frame errors */
  }
}

onUnmounted(stopCamera);
</script>

<template>
  <div>
    <h2>Live Camera Detection</h2>

    <el-card style="margin-bottom: 16px">
      <el-row :gutter="16" align="middle">
        <el-col :span="6">
          <el-select v-model="selectedModel" placeholder="Select model" style="width: 100%">
            <el-option v-for="m in models" :key="m.name" :label="m.display_name" :value="m.name" />
          </el-select>
        </el-col>
        <el-col :span="6">
          <el-button :type="streaming ? 'danger' : 'primary'" @click="streaming ? stopCamera() : startCamera()">
            {{ streaming ? "Stop Camera" : "Start Camera" }}
          </el-button>
        </el-col>
        <el-col :span="6">
          <el-tag v-if="streaming" type="success"
            >Detected: {{ count }} targets | {{ inferenceMs }}ms | {{ fps }} FPS</el-tag
          >
          <el-tag v-else type="info">Camera inactive</el-tag>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="16">
        <el-card header="Live Feed">
          <div style="position: relative; background: #000; min-height: 360px; text-align: center; border-radius: 4px">
            <video ref="videoRef" style="display: none" playsinline />
            <canvas ref="canvasRef" style="max-width: 100%; max-height: 500px" />
            <el-empty
              v-if="!streaming"
              description="Click [Start Camera] to begin live detection"
              style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%)"
            />
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card header="Detected Targets" style="max-height: 500px; overflow-y: auto">
          <div v-if="detections.length === 0" style="text-align: center; color: #909399; padding: 20px">
            {{ streaming ? "No targets in current frame" : "Waiting for camera..." }}
          </div>
          <div
            v-for="(d, i) in detections"
            :key="i"
            style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee"
          >
            <el-tag size="small" :type="d.confidence > 0.7 ? 'success' : d.confidence > 0.5 ? 'warning' : 'danger'">
              {{ d.class_name }}
            </el-tag>
            <span>{{ (d.confidence * 100).toFixed(1) }}%</span>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>
