import { NextResponse } from "next/server"

export async function GET() {
  const data = {
    id: "evt-1",
    date: new Date().toLocaleDateString(),
    time: new Date().toLocaleTimeString(),
    location: "South Entrance",
    summary: "Detected individual matching database profile near restricted area. Escalated to on-site personnel.",
  }
  return NextResponse.json(data)
}
