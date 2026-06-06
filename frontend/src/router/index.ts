import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: () => import('../views/Login.vue'),
    },
    {
      path: '/register',
      name: 'Register',
      component: () => import('../views/Register.vue'),
    },
    {
      path: '/',
      component: () => import('../views/Layout.vue'),
      redirect: '/detect',
      children: [
        {
          path: 'detect',
          name: 'Detect',
          component: () => import('../views/Detect.vue'),
        },
        {
          path: 'history',
          name: 'History',
          component: () => import('../views/History.vue'),
        },
        {
          path: 'video',
          name: 'Video',
          component: () => import('../views/VideoDetect.vue'),
        },
        {
          path: 'camera',
          name: 'Camera',
          component: () => import('../views/Camera.vue'),
        },
        {
          path: 'dataset',
          name: 'Dataset',
          component: () => import('../views/DatasetManager.vue'),
        },
        {
          path: 'eval',
          name: 'Eval',
          component: () => import('../views/ModelEval.vue'),
        },
        {
          path: 'admin',
          name: 'Admin',
          component: () => import('../views/Admin.vue'),
        },
      ],
    },
  ],
})

router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('token')
  if (to.path !== '/login' && to.path !== '/register' && !token) {
    next('/login')
  } else {
    next()
  }
})

export default router
