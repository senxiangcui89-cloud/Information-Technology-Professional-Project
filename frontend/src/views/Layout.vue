<script setup lang="ts">
import { useRouter, RouterView } from "vue-router";
import { useAuthStore } from "../stores/auth";

const router = useRouter();
const auth = useAuthStore();

function handleLogout() {
  auth.logout();
  router.push("/login");
}
</script>

<template>
  <el-container style="min-height: 100vh">
    <el-aside width="220px" style="background: #001529">
      <div style="padding: 20px; text-align: center; color: #fff; font-size: 16px; font-weight: bold">
        Debris Detection
      </div>
      <el-menu
        :default-active="router.currentRoute.value.path"
        background-color="#001529"
        text-color="#ffffff90"
        active-text-color="#fff"
        router
      >
        <el-menu-item index="/detect">
          <el-icon><PictureFilled /></el-icon>
          <span>Image Detection</span>
        </el-menu-item>
        <el-menu-item index="/video">
          <el-icon><VideoCameraFilled /></el-icon>
          <span>Video Detection</span>
        </el-menu-item>
        <el-menu-item index="/camera">
          <el-icon><Camera /></el-icon>
          <span>Live Camera</span>
        </el-menu-item>
        <el-menu-item index="/dataset">
          <el-icon><FolderOpened /></el-icon>
          <span>Dataset Manager</span>
        </el-menu-item>
        <el-menu-item index="/eval">
          <el-icon><DataAnalysis /></el-icon>
          <span>Model Eval</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>History</span>
        </el-menu-item>
        <el-menu-item v-if="auth.isAdmin" index="/admin">
          <el-icon><Setting /></el-icon>
          <span>Users</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header style="display: flex; align-items: center; justify-content: flex-end; border-bottom: 1px solid #eee">
        <span style="margin-right: 16px">{{ auth.username }}</span>
        <el-button type="danger" text @click="handleLogout">Logout</el-button>
      </el-header>
      <el-main style="background: #f5f7fa">
        <RouterView />
      </el-main>
    </el-container>
  </el-container>
</template>
