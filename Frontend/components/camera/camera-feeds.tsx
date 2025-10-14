"use client"

import { CameraFeedCard } from "./camera-feed-card"

export function CameraFeeds() {
  return (
    <div className="grid grid-cols-1 gap-6">
      <CameraFeedCard title="Entrance Camera" cameraId="1" />
    </div>
  )
}
