import type React from "react"
import { Sidebar } from "./sidebar"

export default function PageShell({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="min-h-dvh grid grid-cols-1 md:grid-cols-[72px_1fr]">
      <Sidebar />
      <main className="p-4 md:p-6">
        <header className="mb-4 md:mb-6">
          <h1 className="text-balance text-2xl md:text-3xl font-semibold">{title}</h1>
        </header>
        {children}
      </main>
    </div>
  )
}
