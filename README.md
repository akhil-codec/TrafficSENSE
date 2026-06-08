#  Vehicle Tracking & Speed Detection

A real-time traffic monitoring system that detects, tracks, and measures vehicle speeds from video footage using **YOLOv8** for object detection and **DeepSORT** for multi-object tracking. Generates annotated output video and CSV reports.

---

##  Features

- **Real-time vehicle detection** — identifies Cars, Motorcycles, Buses, and Trucks using YOLOv8
- **Multi-object tracking** — persistent vehicle IDs across frames via DeepSORT
- **Speed estimation** — converts pixel displacement to km/h with configurable calibration
- **Overspeed alerts** — visual warnings when vehicles exceed the configured speed limit
- **Visual overlays** — bounding boxes, centroid trails, speed gauge arcs, and a live stats panel
- **CSV reports** — per-frame data and a per-vehicle summary exported automatically

---

##  Requirements

### Python Version
Python 3.8 or higher

### Dependencies

```bash
pip install ultralytics deep-sort-realtime opencv-python numpy
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model inference |
| `deep-sort-realtime` | DeepSORT tracker |
| `opencv-python` | Video I/O and drawing |
| `numpy` | Numerical operations |

### Model File

Download the YOLOv8 nano weights (auto-downloaded on first run by `ultralytics`):

```bash
# Happens automatically, or manually:
from ultralytics import YOLO
YOLO("yolov8n.pt")
```

---

##  Getting Started

1. **Clone or copy** the script into your project folder.

2. **Place your traffic video** in the same folder and update the path:
   ```python
   VIDEO_PATH = "traffic.mp4"
   ```

3. **Calibrate pixels-per-meter** for your specific camera/scene (see [Calibration](#-calibration) below).

4. **Run the script:**
   ```bash
   python vehicle_tracking_speed.py
   ```

5. Press **Q** in the display window to stop early.

---

##  Configuration

All tunable parameters are at the top of the script:

```python
# ── Files ───────────────────────────────────────────────
VIDEO_PATH          = "traffic_surveillance_video_data.mp4"       # Input video
OUTPUT_VIDEO        = "output_tracking_speed.mp4"
CSV_PER_FRAME       = "vehicle_speeds.csv"
CSV_SUMMARY         = "vehicle_speed_summary.csv"
MODEL_PATH          = "yolov8n.pt"

# ── Speed Estimation ────────────────────────────────────
PIXELS_PER_METER    = 8.0       # MUST be calibrated for your video
SPEED_LIMIT_KMH     = 60.0      # Overspeed threshold
SPEED_SMOOTH_FRAMES = 7         # Moving-average window size

# ── Detection ───────────────────────────────────────────
CONFIDENCE_THRESH   = 0.40      # YOLO confidence threshold

# ── Tracking ────────────────────────────────────────────
DEEPSORT_MAX_AGE    = 30        # Frames to retain a lost track
DEEPSORT_N_INIT     = 3         # Detections to confirm a new track
DEEPSORT_MAX_COSINE = 0.4       # Re-ID appearance similarity threshold

# ── Display ─────────────────────────────────────────────
TRAIL_LENGTH        = 40        # Centroid trail history (frames)
SHOW_WINDOW         = True      # Show live preview window
```

---

##  Calibration

The `PIXELS_PER_METER` value is critical for accurate speed readings. To calibrate:

1. Identify an object of **known real-world length** in the video (e.g., a standard lane is ~3.5 m, a car is ~4.5 m).
2. Measure its **pixel length** in the video frame using an image editor or OpenCV.
3. Calculate:

```
PIXELS_PER_METER = pixel_length / real_world_length_in_meters
```

> **Example:** A car appears 36 px wide and is ~4.5 m long → `36 / 4.5 = 8.0 px/m`

>  Speed accuracy also depends on camera angle. Overhead/top-down angles give the best results.

---

##  Outputs

### 1. Annotated Video (`output_tracking_speed.mp4`)

Each vehicle is rendered with:
- Colored **bounding box** (green = normal, red = overspeeding)
- **Track ID** and vehicle class label
- **Speed** in km/h (with `[OVERSPEED]` flag if exceeded)
- **Centroid trail** showing recent movement path
- **Speed gauge arc** around the centroid
- Red **flash overlay** on overspeeding vehicles
- Live **stats panel** (frame, time, tracked count, overspeed count)
- Full-width **overspeed banner** when any vehicle is over the limit

### 2. Per-Frame CSV (`vehicle_speeds.csv`)

One row per tracked vehicle per frame:

| Column | Description |
|---|---|
| `frame` | Frame number |
| `timestamp_sec` | Time in seconds |
| `track_id` | DeepSORT track ID |
| `vehicle_class` | Car / Motorcycle / Bus / Truck |
| `bbox_x1/y1/x2/y2` | Bounding box coordinates |
| `centroid_x/y` | Box centroid |
| `instant_speed_kmh` | Raw frame-to-frame speed |
| `smooth_speed_kmh` | Exponentially smoothed speed |
| `overspeed_flag` | `1` if over limit, else `0` |

### 3. Summary CSV (`vehicle_speed_summary.csv`)

One row per unique vehicle:

| Column | Description |
|---|---|
| `track_id` | Vehicle ID |
| `vehicle_class` | Detected class |
| `first_frame` / `last_frame` | Visibility range |
| `duration_sec` | Time visible in video |
| `max_speed_kmh` | Peak recorded speed |
| `avg_speed_kmh` | Mean speed over track lifetime |
| `min_speed_kmh` | Lowest recorded speed |
| `overspeed_flag` | `1` if ever exceeded limit |
| `overspeed_note` | e.g. `"Exceeded limit by 12.4 km/h"` |

---

##  Console Output

Progress is printed every 30 frames:

```
  Frame   120/1800  |  Tracked=5  |  Overspeeding=1  |  Proc-FPS=18.3
```

Final summary is printed on completion:

```
═════════════════════════════════════════════════════════════════
  Output video      : output_tracking_speed.mp4
  Per-frame CSV     : vehicle_speeds.csv
  Summary CSV       : vehicle_speed_summary.csv
  Total frames      : 1800
  Unique vehicle IDs: 23
  Overspeeding IDs  : 4  →  [3, 7, 12, 19]
  Fastest vehicle   : ID 7  (Car)  @  84.3 km/h
═════════════════════════════════════════════════════════════════
```

---

##  Project Structure

```
project/
├── vehicle_tracking_speed.py          # Main script
├── traffic_surveillance_video_data.mp4                 # Input video (user-provided)
├── yolov8n.pt                  # YOLOv8 weights (auto-downloaded)
├── output_tracking_speed.mp4   # Annotated output video (generated)
├── vehicle_speeds.csv          # Per-frame data (generated)
└── vehicle_speed_summary.csv   # Per-vehicle summary (generated)
```

---

##  Performance Tips

- Use **`yolov8n.pt`** (nano) for faster inference on CPU; switch to `yolov8s.pt` or larger for better accuracy.
- Set `SHOW_WINDOW = False` to skip the display window and improve processing speed.
- Enable GPU by ensuring a CUDA-compatible PyTorch build is installed — `ultralytics` will use it automatically.
- Reduce `TRAIL_LENGTH` and `SPEED_SMOOTH_FRAMES` if RAM usage is a concern for long videos.

---

##  Known Limitations

- Speed accuracy degrades at steep camera angles; best results with near-overhead views.
- `PIXELS_PER_METER` assumes a **flat road plane** — perspective distortion is not corrected.
- Track IDs may reset or swap during occlusions (inherent DeepSORT limitation).
- Night or low-contrast footage may reduce YOLO detection confidence.

---


