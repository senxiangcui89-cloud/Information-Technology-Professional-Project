export interface ModelInfo {
  name: string
  display_name: string
  params: string
  mAP50: number | null
  mAP50_95: number | null
  fps: number | null
}

export interface DetectionBox {
  class_name: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
}
