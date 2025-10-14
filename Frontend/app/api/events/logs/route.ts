import { NextResponse } from "next/server"

export async function GET() {
  const now = Date.now()
  const logs = [
    {
      id: "l-1",
      timestamp: new Date(now - 1000 * 120).toISOString(),
      message: "Badge access denied at South Entrance.",
    },
    {
      id: "l-2",
      timestamp: new Date(now - 1000 * 60).toISOString(),
      message: "Security notified of suspicious activity.",
    },
    {
      id: "l-3",
      timestamp: new Date(now - 1000 * 10).toISOString(),
      message: "Subject left the restricted area.",
    },
  ]
  return NextResponse.json(logs)
}
