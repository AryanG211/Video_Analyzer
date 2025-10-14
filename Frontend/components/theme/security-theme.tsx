"use client"

import type * as React from "react"

type Props = { children: React.ReactNode }

/**
 * SecurityTheme
 * Applies a security-agency oriented palette via CSS variables.
 * Navy primary, slate neutrals, teal accent. Keeps shadcn tokens.
 */
export default function SecurityTheme({ children }: Props) {
  const vars = {
    // Core palette
    // Deep navy backgrounds with slate foregrounds, teal ring/accent
    ["--background" as any]: "#0b1220",
    ["--foreground" as any]: "#e5e7eb",

    ["--card" as any]: "#0f1a2a",
    ["--card-foreground" as any]: "#e5e7eb",

    ["--muted" as any]: "#162233",
    ["--muted-foreground" as any]: "#94a3b8",

    ["--popover" as any]: "#0f1a2a",
    ["--popover-foreground" as any]: "#e5e7eb",

    ["--primary" as any]: "#0a3a66", // navy
    ["--primary-foreground" as any]: "#e6f1ff",

    ["--secondary" as any]: "#1b2a3d",
    ["--secondary-foreground" as any]: "#cbd5e1",

    ["--accent" as any]: "#14b8a6", // teal
    ["--accent-foreground" as any]: "#062b3a",

    ["--destructive" as any]: "#ef4444",
    ["--destructive-foreground" as any]: "#1b0b0b",

    ["--border" as any]: "#233048",
    ["--input" as any]: "#233048",
    ["--ring" as any]: "#14b8a6",

    ["--radius" as any]: "0.5rem",
  } as React.CSSProperties

  return (
    <div style={vars} className="min-h-screen bg-background text-foreground">
      {children}
    </div>
  )
}
