import { NextResponse } from "next/server"

export async function GET() {
  const now = Date.now()
  const data = [
    {
      id: "a-1",
      timestamp: new Date(now - 1000 * 60).toISOString(),
      cameraId: "cam-1",
      level: "warning",
      message: "Motion detected near south entrance.",
    },
    {
      id: "a-2",
      timestamp: new Date(now - 1000 * 30).toISOString(),
      cameraId: "cam-2",
      level: "info",
      message: "Camera 2 resumed streaming.",
    },
    {
      id: "a-3",
      timestamp: new Date(now - 1000 * 10).toISOString(),
      cameraId: "cam-1",
      level: "critical",
      message: "Unidentified person loitering for > 3 minutes.",
    },
  ]
  return NextResponse.json(data)
}
