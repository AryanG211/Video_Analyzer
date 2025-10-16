<h1 align="center">NETRADOOT - AI Live Footage Analyzer</h1>  

#### We are Team Krakenova and here's the glimpse of our solution.

---

## Project Structure  

### Backend  

| File / Folder | Description |
|----------------|-------------|
| `.gitignore` | Specifies intentionally untracked files to ignore in Git version control |
| `api_server.py` | Handles database page and report downloading function |
| `best (3).pt` | Pre-trained YOLO model weights used for object or weapon detection |
| `detection_server.py` | Handles video frame processing and runs detection algorithms |
| `merger.py` | Merges and integrates results from multiple detection modules for unified output |
| `report.py` | Generates detailed analysis reports (e.g., detections, timestamps, summaries) |
| `requirements.txt` | Lists all Python dependencies required to run the backend |
| `yolov8n-pose.pt` | YOLOv8 Pose model weights for human pose estimation tasks |

---

### Frontend  

| File / Folder | Description |
|----------------|-------------|
| `app/` | Core Next.js app directory containing main routes, pages, and layouts |
| `components/` | Reusable UI components like buttons, cards, modals, and forms |
| `hooks/` | Custom React hooks to modularize reusable logic |
| `lib/` | Helper functions, utilities, and configuration scripts |
| `public/` | Static assets including images, icons, and media files |
| `styles/` | Global CSS, Tailwind, and styling configuration files |
| `.gitignore` | Specifies files and folders ignored by Git for the frontend |
| `components.json` | Configuration file for UI component definitions |
| `next.config.mjs` | Main configuration file for Next.js app settings |
| `package-lock.json` | Locked npm dependencies ensuring consistent installs |
| `package.json` | Project metadata, scripts, and dependency definitions for the frontend |
| `pnpm-lock.yaml` | Lock file if using pnpm package manager |
| `postcss.config.mjs` | Configuration file for PostCSS (used with TailwindCSS) |
| `tsconfig.json` | TypeScript configuration defining compiler and project settings |
