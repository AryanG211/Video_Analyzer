"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"

type RawAlert = {
  type: string
  name?: string
  similarity?: number | string
  confidence?: number | string
  activity?: string
  id?: string
  speed?: number | string
}

type AlertData = {
  frame: number
  timestamp: number
  id: number
  center?: [number, number]
  bbox?: [number, number, number, number]
  alerts: string[]
}

export function AlertLog() {
  const [logs, setLogs] = useState<RawAlert[]>([])
  const [connectionStatus, setConnectionStatus] = useState<
    "connected" | "disconnected" | "error"
  >("disconnected")

  useEffect(() => {
    let intervalId: NodeJS.Timeout

    async function fetchData() {
      try {
        const res = await fetch("/alerts.json")
        const alertsData: AlertData[] = await res.json()

        if (!alertsData || alertsData.length === 0) return

        // 🧹 Clean invalid entries
        const cleanedData = alertsData.filter(
          a =>
            a &&
            typeof a.timestamp === "number" &&
            Array.isArray(a.center) &&
            a.center.length === 2
        )

        // 🧠 Group by ID
        const groups: { [key: number]: AlertData[] } = {}
        cleanedData.forEach(item => {
          if (!groups[item.id]) groups[item.id] = []
          groups[item.id].push({ ...item, speed: 0 })
        })

        // 🚀 Compute speed safely
        Object.keys(groups).forEach(id => {
          const track = groups[Number(id)]
          track.sort((a, b) => a.timestamp - b.timestamp)
          for (let i = 1; i < track.length; i++) {
            const prev = track[i - 1]
            const curr = track[i]
            if (
              curr &&
              prev &&
              Array.isArray(curr.center) &&
              Array.isArray(prev.center) &&
              curr.center.length === 2 &&
              prev.center.length === 2 &&
              curr.timestamp > prev.timestamp
            ) {
              const dx = curr.center[0] - prev.center[0]
              const dy = curr.center[1] - prev.center[1]
              const dist = Math.sqrt(dx * dx + dy * dy)
              const dt = curr.timestamp - prev.timestamp
              curr.speed = dist / dt
            } else {
              curr.speed = 0
            }
          }
        })

        // 🔁 Flatten & sort
        const sortedAlerts = cleanedData
          .map(item => ({
            ...item,
            speed:
              groups[item.id].find(
                t => t.timestamp === item.timestamp && t.frame === item.frame
              )?.speed || 0
          }))
          .sort((a, b) => a.timestamp - b.timestamp)

        if (sortedAlerts.length === 0) return
        setConnectionStatus("connected")

        const minTime = sortedAlerts[0].timestamp
        const playbackSpeed = 4 // smaller = slower
        const startTime = Date.now()
        const alertsQueue = [...sortedAlerts]

        intervalId = setInterval(() => {
          const elapsed = ((Date.now() - startTime) / 1000) * playbackSpeed
          const newLogs: RawAlert[] = []

          while (
            alertsQueue.length > 0 &&
            (alertsQueue[0].timestamp - minTime) <= elapsed
          ) {
            const item = alertsQueue.shift()!
            const alert: RawAlert = {
              type: item.alerts.includes("RUN")
                ? "RUNNING"
                : item.alerts.includes("LOITERING")
                ? "LOITERING"
                : "ACTIVITY",
              activity: item.alerts.join(", "),
              id: item.id.toString(),
              speed: item.speed
            }
            newLogs.push(alert)
          }

          if (newLogs.length > 0) {
            setLogs(prev => [...newLogs, ...prev].slice(0, 100))
          }

          if (alertsQueue.length === 0) clearInterval(intervalId)
        }, 200)
      } catch (err) {
        console.error("Error loading alerts.json:", err)
        setConnectionStatus("error")
      }
    }

    fetchData()

    return () => clearInterval(intervalId)
  }, [])

  const formatAlert = (alert: RawAlert) => {
    switch (alert.type) {
      case "FACE_MATCH":
        return `Face matched: ${alert.name} (Similarity: ${Number(alert.similarity).toFixed(2)})`
      case "WEAPON":
        return `Weapon detected: ${alert.name || "Unknown"} (Confidence: ${Number(alert.confidence).toFixed(2)})`
      case "RUNNING":
      case "LOITERING":
      case "ACTIVITY":
        return `Activity: ${alert.activity} | ID: ${alert.id} | Speed: ${Number(alert.speed).toFixed(2)} px/s`
      default:
        return alert.name || JSON.stringify(alert)
    }
  }

  const getBorderColor = (alert: RawAlert) => {
    switch (alert.type) {
      case "FACE_MATCH":
        return "border-blue-500 shadow-[0_0_10px_2px_rgba(59,130,246,0.7)]"
      case "WEAPON":
        return "border-red-500 shadow-[0_0_10px_2px_rgba(239,68,68,0.7)]"
      case "RUNNING":
      case "LOITERING":
      case "ACTIVITY":
        return "border-yellow-500 shadow-[0_0_10px_2px_rgba(234,179,8,0.7)]"
      default:
        return "border-gray-300"
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Live Alerts</CardTitle>
        <span
          className={`text-sm ${
            connectionStatus === "connected"
              ? "text-green-600"
              : connectionStatus === "error"
              ? "text-red-600"
              : "text-yellow-600"
          }`}
        >
          {connectionStatus === "connected"
            ? "Connected (Simulated)"
            : connectionStatus === "error"
            ? "Connection Error"
            : "Disconnected"}
        </span>
      </CardHeader>

      <CardContent>
        <ScrollArea className="h-64">
          <ul className="grid gap-2 pr-3">
            {logs.length > 0 ? (
              logs.map((alert, idx) => (
                <li
                  key={idx}
                  className={`rounded-md border p-3 text-sm bg-gray-50 font-mono transition-all duration-300 ${getBorderColor(alert)}`}
                >
                  {formatAlert(alert)}
                </li>
              ))
            ) : (
              <li className="text-muted-foreground text-sm">No alerts yet.</li>
            )}
          </ul>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

