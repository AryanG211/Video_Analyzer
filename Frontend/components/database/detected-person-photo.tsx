"use client"

import useSWR from "swr"
import Image from "next/image"
import { Card } from "@/components/ui/card"
import { fetcher } from "@/lib/swr-fetcher"

type DetectedPerson = {
  id: string | number
  photoUrl: string
  timestamp: string
  cameraId: string | number
}

export function DetectedPersonWithEvent() {
  // Fetch detected people
  const { data: people, error: peopleError, isLoading: peopleLoading } = useSWR<DetectedPerson[]>(
    "http://localhost:8001/api/people",
    fetcher,
    { revalidateOnFocus: false, refreshInterval: 5000 }
  )

  if (peopleError) return <p className="text-red-500">Error loading data.</p>
  if (peopleLoading) return <p>Loading data...</p>
  if (!people) return <p>No data available.</p>

  return (
    <div className="space-y-4">
      {people.map((person) => (
        <Card key={person.id} className="h-48 w-48 p-2">
          {/* Image */}
          <div className="w-43 h-44 relative flex-shrink-0 rounded-lg ring-1 ring-border bg-muted">
            <Image
              src={person.photoUrl}
              alt={`Detected person ${person.id}`}
              fill
              className="object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-background/80 backdrop-blur px-1 py-0.5 text-[10px] truncate">
              Captured: {new Date(person.timestamp).toLocaleString()} • Camera {person.cameraId}
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}

