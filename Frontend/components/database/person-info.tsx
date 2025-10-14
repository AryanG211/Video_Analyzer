// "use client"

// import useSWR from "swr"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Badge } from "@/components/ui/badge"
// import { fetcher } from "@/lib/swr-fetcher"

// type Person = {
//   id: string
//   name: string
//   gender: string
//   age: number
//   classified: boolean
//   wanted: boolean
//   firstTimeOffender: boolean
//   crimes: string[]
// }

// export function PersonInfo() {
//   const { data } = useSWR<Person>("/api/people", fetcher, {
//     revalidateOnFocus: false,
//   })

//   return (
//     <Card>
//       <CardHeader>
//         <CardTitle className="text-pretty">Person Information</CardTitle>
//       </CardHeader>
//       <CardContent className="grid gap-3 text-sm">
//         <div className="flex flex-wrap items-center gap-2">
//           <span className="text-muted-foreground">Name:</span>
//           <span className="font-medium">{data?.name || "—"}</span>
//           {data?.classified ? (
//             <Badge variant="secondary">Classified</Badge>
//           ) : (
//             <Badge variant="outline">Unclassified</Badge>
//           )}
//           {data?.wanted ? <Badge className="bg-destructive text-destructive-foreground">Wanted</Badge> : null}
//           {data?.firstTimeOffender ? <Badge variant="outline">First-time</Badge> : null}
//         </div>
//         <div className="grid grid-cols-2 gap-3">
//           <div>
//             <span className="text-muted-foreground">Gender:</span> {data?.gender || "—"}
//           </div>
//           <div>
//             <span className="text-muted-foreground">Age:</span> {data?.age ?? "—"}
//           </div>
//         </div>
//         <div className="flex flex-wrap gap-2">
//           <span className="text-muted-foreground">Crimes:</span>
//           <div className="flex flex-wrap gap-2">
//             {(data?.crimes?.length ? data.crimes : ["—"]).map((c) => (
//               <Badge key={c} variant="outline">
//                 {c}
//               </Badge>
//             ))}
//           </div>
//         </div>
//       </CardContent>
//     </Card>
//   )
// }
