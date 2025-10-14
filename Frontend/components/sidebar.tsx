"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Camera, Database, Menu } from "lucide-react"
import { cn } from "@/lib/utils"

export function Sidebar() {
  const pathname = usePathname()
  const linkBase = "flex items-center justify-center h-10 w-10 rounded-md border transition-colors"
  const active =
    "bg-(--color-sidebar-primary) text-(--color-sidebar-primary-foreground) border-(--color-sidebar-border)"
  const idle =
    "bg-(--color-sidebar) text-(--color-sidebar-foreground) hover:bg-(--color-sidebar-accent) border-(--color-sidebar-border)"

  return (
    <aside className="hidden md:flex flex-col items-center gap-4 p-3 border-r bg-sidebar" aria-label="Primary">
      <button
        aria-label="Open menu"
        className="h-10 w-10 grid place-items-center rounded-md border bg-(--color-sidebar) text-(--color-sidebar-foreground)"
      >
        <Menu className="h-5 w-5" />
      </button>

      <nav className="flex flex-col gap-2" aria-label="Main">
        <Link
          href="/camera"
          aria-current={pathname.startsWith("/camera") ? "page" : undefined}
          className={cn(linkBase, pathname.startsWith("/camera") ? active : idle)}
        >
          <Camera className="h-5 w-5" />
        </Link>
        <Link
          href="/database"
          aria-current={pathname.startsWith("/database") ? "page" : undefined}
          className={cn(linkBase, pathname.startsWith("/database") ? active : idle)}
        >
          <Database className="h-5 w-5" />
        </Link>
      </nav>
    </aside>
  )
}
