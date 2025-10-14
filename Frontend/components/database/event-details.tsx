"use client"

import useSWR from "swr"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MapPin, CalendarClock } from "lucide-react"
import { fetcher } from "@/lib/swr-fetcher"

type Event = {
  id: string
  date: string
  time: string
  location: string
  summary: string
  personId?: string
  weaponId?: string
}

export function EventDetails() {
  const { data, error, isLoading } = useSWR<Event[]>("http://localhost:8001/api/events", fetcher, {
    revalidateOnFocus: false,
    refreshInterval: 5000,
  })

  console.log("EVENT DATA FROM API:", data)

  if (error) return <p className="text-red-500">Error loading events.</p>
  if (isLoading) return <p>Loading events...</p>
  if (!Array.isArray(data) || data.length === 0) return <p>No events yet.</p>

  return (
    <div className="w-full flex justify-end pr-10">
      <div className="space-y-4 flex flex-col items-end">
        {data.map(event => (
          <Card
            key={event.id}
            className="relative h-48 w-150 transform -translate-x-100 transition-transform duration-300 shadow-md"
          >
            <CardHeader className="flex items-center justify-between space-y-0 p-2">
              <CardTitle className="text-pretty text-xs font-semibold">Event Details</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-1 text-xs p-2">
              <div className="flex items-center gap-1">
                <CalendarClock className="h-3 w-3 text-muted-foreground" aria-hidden />
                <span>
                  <strong className="text-foreground">{event.date}</strong> at{" "}
                  <strong className="text-foreground">{event.time}</strong>
                </span>
              </div>
              <div className="flex items-center gap-1">
                <MapPin className="h-3 w-3 text-muted-foreground" aria-hidden />
                <span className="truncate">{event.location}</span>
              </div>
              <p className="text-pretty leading-tight line-clamp-2">{event.summary}</p>
              {event.personId && <p className="truncate">Person ID: {event.personId}</p>}
              {event.weaponId && <p className="truncate">Weapon ID: {event.weaponId}</p>}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

