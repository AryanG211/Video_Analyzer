// import PageShell from "@/components/page-shell"
// import { CameraFeedCard } from "@/components/camera/camera-feed-card"
// import { AlertLog } from "@/components/camera/alert-log"
// import { Heatmap } from "@/components/camera/heatmap"

// const cameras = [
//   { id: "cam-1", title: "Camera Feed 1" },
//   { id: "cam-2", title: "Camera Feed 2" },
//   // Future: add more camera objects here; the mapping components already handle >2 feeds
// ]

// export default function CameraPage() {  
//   return (
//     <PageShell title="Camera feed page">
//       <section className="grid gap-6">
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//           <CameraFeedCard title={cameras[0].title} cameraId={cameras[0].id} />
//           <CameraFeedCard title={cameras[1].title} cameraId={cameras[1].id} />
//         </div>
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//           <AlertLog />
//           <Heatmap />
//         </div>
//       </section>
//     </PageShell>
//   )
// }
import PageShell from "@/components/page-shell"
import { CameraFeedCard } from "@/components/camera/camera-feed-card"
import { AlertLog } from "@/components/camera/alert-log"
import { Heatmap } from "@/components/camera/heatmap"

const cameras = [
  { id: "cam-1", title: "Camera Feed 1" },
]

export default function CameraPage() {  
  return (
    <PageShell title="Camera feed page">
      <section className="grid gap-6">
        {/* Camera Feed Section */}
        <div className="flex justify-center">
          <div className="w-full md:w-[600px]">
            <CameraFeedCard title={cameras[0].title} cameraId={cameras[0].id} />
          </div>
        </div>

        {/* Alert + Heatmap Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <AlertLog />
          <Heatmap />
        </div>
      </section>
    </PageShell>
  )
}

