import { NextResponse } from "next/server"

export async function GET() {
  const person = {
    id: "p-1",
    name: "Sample Person",
    gender: "Unknown",
    age: 34,
    classified: true,
    wanted: false,
    firstTimeOffender: false,
    crimes: ["Trespassing"],
    photoUrl: "/detected-person-photo.jpg",
  }
  return NextResponse.json(person)
}
