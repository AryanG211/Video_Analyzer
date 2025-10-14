// "use client"

// import useSWR from "swr"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { ScrollArea } from "@/components/ui/scroll-area"
// import { fetcher } from "@/lib/swr-fetcher"

// type Log = {
//   id: string
//   timestamp: string
//   message: string
// }

// export function LogEntries() {
//   const { data } = useSWR<Log[]>("/api/events/logs", fetcher, {
//     revalidateOnFocus: false,
//   })

//   return (
//     <Card>
//       <CardHeader>
//         <CardTitle className="text-pretty">Log entries</CardTitle>
//       </CardHeader>
//       <CardContent>
//         <ScrollArea className="h-64">
//           <ul className="grid gap-2 pr-3">
//             {data?.map((l) => (
//               <li
//                 key={l.id}
//                 className="rounded-md border p-3 text-sm bg-card text-card-foreground pl-4 border-l-4 border-l-accent"
//               >
//                 <time dateTime={l.timestamp} className="text-muted-foreground">
//                   {new Date(l.timestamp).toLocaleString()}
//                 </time>
//                 <div className="mt-1 text-pretty">{l.message}</div>
//               </li>
//             )) || <li className="text-muted-foreground">No logs.</li>}
//           </ul>
//         </ScrollArea>
//       </CardContent>
//     </Card>
//   )
// }
