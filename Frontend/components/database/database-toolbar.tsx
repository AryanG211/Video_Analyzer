"use client"

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Search, Filter, Download } from "lucide-react"

export function DatabaseToolbar() {
  const [q, setQ] = useState("")
  const [status, setStatus] = useState<string | undefined>()
  const [classification, setClassification] = useState<string | undefined>()

  return (
    <div className="mb-4 md:mb-6 grid grid-cols-1 md:grid-cols-[1fr_auto_auto_auto] gap-3">
      <div className="flex items-center gap-2">
        <Search className="h-4 w-4 text-muted-foreground" aria-hidden />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search name, ID, location…"
          aria-label="Search database"
        />
      </div>
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" aria-hidden />
        <Select onValueChange={(v) => setStatus(v)} value={status}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="wanted">Wanted</SelectItem>
            <SelectItem value="cleared">Cleared</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Select onValueChange={(v) => setClassification(v)} value={classification}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Classification" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="classified">Classified</SelectItem>
          <SelectItem value="unclassified">Unclassified</SelectItem>
        </SelectContent>
      </Select>
      <div className="flex justify-end gap-2">
        <Button variant="secondary" className="w-full md:w-auto" aria-label="Apply filters">
          Apply
        </Button>
        <Button
            asChild
            variant="secondary"
            className="w-full md:w-auto"
            aria-label="Download reports"
          >
            <a href="http://localhost:8001/api/reports" download="events-report.pdf">
              <Download className="h-4 w-4 mr-2" aria-hidden />
              Download Reports
            </a>
        </Button>

      </div>
    </div>
  )
}