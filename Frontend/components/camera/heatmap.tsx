"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function Heatmap() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-pretty">Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <Button
          variant="default"
          className="w-full bg-gradient-to-r from-orange-400 to-red-500 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:from-orange-500 hover:to-red-600 transition-all duration-300"
        >
          Enable Heatmap
        </Button>
      </CardContent>
    </Card>
  )
}
