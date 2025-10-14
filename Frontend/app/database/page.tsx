import PageShell from "@/components/page-shell"
import { DetectedPersonWithEvent } from "@/components/database/detected-person-photo"
import { EventDetails } from "@/components/database/event-details"
// import { PersonInfo } from "@/components/database/person-info"
// import { LogEntries } from "@/components/database/log-entries"
import { DatabaseToolbar } from "@/components/database/database-toolbar"

export default function DatabasePage() {
  return (
    <PageShell title="Database page">
      <DatabaseToolbar />
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DetectedPersonWithEvent />
        <EventDetails />
        {/* <PersonInfo />
        <LogEntries /> */}
      </section>
    </PageShell>
  )
}
