export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const eventId = searchParams.get("eventId") || "evt-1"

  const rows = [
    ["event_id", "date", "time", "location", "summary"],
    [
      eventId,
      new Date().toLocaleDateString(),
      new Date().toLocaleTimeString(),
      "South Entrance",
      "Detected individual near restricted area; report auto-generated.",
    ],
  ]
  const csv = rows.map((r) => r.map(escapeCSV).join(",")).join("\n")

  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv",
      "Content-Disposition": `attachment; filename="${eventId}.csv"`,
    },
  })
}

function escapeCSV(val: string) {
  if (/[",\n]/.test(val)) {
    return `"${val.replace(/"/g, '""')}"`
  }
  return val
}
