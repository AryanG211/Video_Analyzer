"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function CameraFeedCard({
  title,
  cameraId,
}: {
  title: string
  cameraId: string
}) {
  const [isOnline, setIsOnline] = useState(true)

  // Check if backend feed is reachable
  useEffect(() => {
    const checkFeed = async () => {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 2000) // 2 sec timeout

        const response = await fetch(
          `http://localhost:8000/video_feed?cameraId=${cameraId}`,
          { method: "GET", signal: controller.signal }
        )

        clearTimeout(timeout)
        setIsOnline(response.ok)
      } catch (error) {
        setIsOnline(false)
      }
    }

    checkFeed()
    const interval = setInterval(checkFeed, 5000) // check every 5 seconds
    return () => clearInterval(interval)
  }, [cameraId])

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex items-center justify-between">
        <CardTitle className="text-pretty">{title}</CardTitle>
        <span
          className={`inline-flex items-center gap-2 text-xs ${
            isOnline ? "text-green-500" : "text-red-500"
          }`}
          aria-label={`Camera status ${isOnline ? "online" : "offline"}`}
          title="Camera status"
        >
          <span
            className={`h-2.5 w-2.5 rounded-full ring-2 ${
              isOnline ? "bg-green-500 ring-green-500/20" : "bg-red-500 ring-red-500/20"
            }`}
          />
          {isOnline ? "Online" : "Offline"}
        </span>
      </CardHeader>

      <CardContent className="p-0">
        <div className="relative aspect-video w-full bg-black">
          {isOnline ? (
            <img
              src={`http://localhost:8000/video_feed?cameraId=${cameraId}`}
              alt={`Live view for ${title}`}
              className="object-cover w-full h-full"
            />
          ) : (
            <div className="flex items-center justify-center w-full h-full text-red-500">
              Feed not available
            </div>
          )}
        </div>
        <div className="p-3 text-sm text-muted-foreground">Camera ID: {cameraId}</div>
      </CardContent>
    </Card>
  )
}

