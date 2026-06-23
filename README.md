# Traffic Surveillance & Risk Monitoring System

A real-time traffic analysis tool that uses YOLOv8 and computer vision to detect vehicles, estimate speeds, flag lane crossings, compute traffic risk scores, and **adaptively control traffic signal timings** — all from video footage.

> **Tip — Run on Google Colab with a T4 GPU**
> Processing traffic video with `yolov8x.pt` at 1280px resolution is compute-intensive. If you don't have a local GPU, **Google Colab's free T4 GPU** is an excellent option that can be 10–20× faster than CPU.
> 1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
> 2. Enable the GPU: **Runtime → Change runtime type → T4 GPU → Save**.
> 3. Upload `traffic_monitor.py`, `adaptive_signal.py`, and your video, then install dependencies and run the script in a cell.
> 4. Download `output.mp4` and the CSV files from the Colab file browser when done.
>
> ```python
> # In a Colab cell:
> !pip install ultralytics supervision opencv-python-headless numpy pandas matplotlib tqdm
> !python traffic_monitor.py
> ```
> Note: use `opencv-python-headless` on Colab since there is no display; the script's `plt.show()` calls can be replaced with `plt.savefig(...)` for plot export.

---

## Features

- **Vehicle Detection & Tracking** — Detects cars, motorcycles, buses, and trucks using YOLOv8, tracked across frames with ByteTrack.
- **Speed Estimation** — Computes per-vehicle speed in km/h using perspective transformation to map pixel coordinates to real-world distances.
- **Overspeed Detection** — Flags and highlights vehicles exceeding a configurable speed limit.
- **Lane Crossing Detection** — Detects illegal lane changes using pre-defined polygon boundaries; logs events and displays on-screen alerts.
- **Traffic Risk Scoring** — Calculates a composite risk score per frame based on vehicle count, speed variance, and congestion level.
- **Congestion Monitoring** — Classifies traffic density as Low, Medium, or High and triggers on-screen warning banners.
- **Adaptive Signal Control** — Dynamically adjusts per-lane green signal durations based on real-time vehicle density, speed variance, and congestion level. Includes emergency override to force green for critically congested lanes.
- **Annotated Video Output** — Produces an output video with bounding boxes, speed labels, trace trails, HUD stats, adaptive signal status, and warning banners.
- **CSV Data Export** — Writes per-frame and per-vehicle summary CSVs for downstream analysis.
- **Post-run Plots** — Generates matplotlib charts of overspeed events, lane crossings, and risk scores over time.

---

## Project Structure

```
.
├── traffic_monitor.py            # Main script
├── adaptive_signal.py            # Adaptive signal controller module
├── traffic_surveillance_video.mp4  # Input video (user-supplied)
├── output.mp4                    # Annotated output video (generated)
├── vehicle_speeds.csv            # Per-frame data (generated)
└── vehicle_speed_summary.csv     # Per-vehicle summary (generated)
```

---

## Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install ultralytics supervision opencv-python numpy pandas matplotlib tqdm
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model inference |
| `supervision` | Tracking, annotation, video I/O |
| `opencv-python` | Frame processing, perspective transform |
| `numpy` | Array math |
| `pandas` | CSV loading for plot generation |
| `matplotlib` | Post-run metric charts |
| `tqdm` | Progress bar |

---

## Setup

### 1. Download the YOLO model

The script uses `yolov8x.pt` by default. It will auto-download on first run if you have `ultralytics` installed. To use a lighter model, change `MODEL_NAME`:

```python
MODEL_NAME = "yolov8n.pt"  # nano — fastest
MODEL_NAME = "yolov8s.pt"  # small
MODEL_NAME = "yolov8x.pt"  # extra-large — most accurate (default)
```

### 2. Prepare your video

Place your traffic video in the project directory and update the path:

```python
SOURCE_VIDEO_PATH = "traffic_surveillance_video.mp4"
```

### 3. Configure the perspective transform

Speed estimation maps a trapezoidal road region in the video to a real-world rectangle. Update `SOURCE` with the four corner pixel coordinates of the road area, and `TARGET_WIDTH` / `TARGET_HEIGHT` with the actual road dimensions in metres:

```python
SOURCE = np.array([
    [781, 216], [1056, 216], [1675, 530], [184, 482]
])
TARGET_WIDTH = 42    # Road width in metres
TARGET_HEIGHT = 210  # Road length in metres
```

To find the correct pixel coordinates, open your video in a tool like VLC (Tools → Media Information → Codec Details shows resolution) or use OpenCV to display the first frame and note the corner pixel positions of the visible road segment.

### 4. Define lane polygons

Lane boundaries are hard-coded as `LANE_POLYGONS` — a list of `np.int32` polygon arrays, one per lane. Each polygon is defined by its corner pixel coordinates on the video frame:

```python
LANE_POLYGONS = [
    np.array([[48, 634], [801, 240], [828, 246], [137, 744]], dtype=np.int32),
    np.array([[137, 744], [828, 244], [861, 254], [312, 870]], dtype=np.int32),
    # ... add one entry per lane
]
```

The number of lanes defined here also sets how many signal phases the adaptive controller manages. A practical way to determine these coordinates is to run the script once on your video, pause on the first frame, and use an image viewer or OpenCV to note the polygon corner positions.

---

## Usage

```bash
python traffic_monitor.py
```

The script starts processing immediately with no GUI interaction required. A progress bar tracks frame-by-frame progress.

---

## Configuration Reference

### `traffic_monitor.py`

All tunable parameters are at the top of `traffic_monitor.py`:

| Parameter | Default | Description |
|---|---|---|
| `SOURCE_VIDEO_PATH` | `traffic_surveillance_video.mp4` | Input video path |
| `TARGET_VIDEO_PATH` | `output.mp4` | Annotated output video path |
| `CSV_PER_FRAME` | `vehicle_speeds.csv` | Per-frame data export |
| `CSV_SUMMARY` | `vehicle_speed_summary.csv` | Per-vehicle summary export |
| `CONFIDENCE_THRESHOLD` | `0.3` | Minimum YOLO detection confidence |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `MODEL_NAME` | `yolov8x.pt` | YOLO model weights |
| `MODEL_RESOLUTION` | `1280` | Inference resolution (pixels) |
| `SPEED_LIMIT` | `60.0` | Speed limit in km/h |
| `HIGH_DENSITY_THRESHOLD` | `10` | Vehicle count to trigger high-density alert |
| `CONGESTION_THRESHOLDS` | `low: 5, medium: 10` | Vehicle count thresholds for congestion levels |
| `RISK_WEIGHTS` | `α=1.2, β=0.4, γ=3.0` | Weights for risk score formula |
| `LANE_POLYGONS` | 6 pre-set polygons | Lane boundary definitions (pixel coordinates) |

### `adaptive_signal.py`

Signal timing parameters are tunable at the top of `adaptive_signal.py`:

| Parameter | Default | Description |
|---|---|---|
| `BASE_GREEN_SEC` | `20.0` | Default green phase duration (seconds) |
| `BASE_RED_SEC` | `15.0` | Default red phase duration (seconds) |
| `MIN_GREEN_SEC` | `8.0` | Minimum green time (floor) |
| `MAX_GREEN_SEC` | `60.0` | Maximum green time (cap, to prevent starvation) |
| `YELLOW_SEC` | `3.0` | Fixed yellow phase duration |
| `DENSITY_WEIGHT` | `1.5` | Extra seconds per vehicle above the low-density threshold |
| `SPEED_VAR_WEIGHT` | `0.4` | Extra seconds per unit of speed standard deviation |
| `CONGESTION_BONUS` | `low: 0, medium: 8, high: 20` | Flat green-time bonus (seconds) by congestion level |
| `LOW_DENSITY_THRESH` | `3` | Vehicle count below which no density bonus is applied |
| `MEDIUM_DENSITY_THRESH` | `7` | Vehicle count threshold separating medium from high density |

---

## Adaptive Signal Control

The adaptive signal system (`adaptive_signal.py`) runs alongside the main detector and adjusts green light durations every frame based on live lane data.

### How It Works

Each lane gets its own independent signal phase cycle: **GREEN → YELLOW → RED → GREEN**. The green duration is recomputed each cycle using:

```
green = BASE_GREEN_SEC
      + DENSITY_WEIGHT  × max(0, vehicle_count − LOW_DENSITY_THRESH)
      + SPEED_VAR_WEIGHT × speed_std_dev_kmh
      + CONGESTION_BONUS[congestion_level]
```

The result is clamped to `[MIN_GREEN_SEC, MAX_GREEN_SEC]`.

### Per-lane Stats Fed to the Controller

Every frame, the main script computes a `LaneStats` snapshot for each lane:

| Field | Description |
|---|---|
| `vehicle_count` | Number of vehicles whose bounding-box centre falls inside the lane polygon |
| `avg_speed_kmh` | Mean speed of vehicles in the lane |
| `speed_std_kmh` | Speed standard deviation (traffic chaos indicator) |
| `congestion_level` | `"low"` / `"medium"` / `"high"` based on vehicle count thresholds |
| `lane_crossing_count` | Active lane-crossing events in this lane |
| `overspeed_count` | Active overspeeding vehicles in this lane |

### Emergency Override

If a lane is classified as `"high"` congestion but its signal is currently RED, the system immediately forces it to GREEN — preventing severe queue build-up:

```python
signal_controller.force_green(lane_id)
```

### Signal Phase Staggering

On startup, lanes are given staggered offsets so they naturally alternate rather than all turning green at once.

### HUD Display

The adaptive signal status is rendered in the video as a live panel below the main stats HUD, showing each lane's current color, countdown timer, progress bar, and the reason for the current green adjustment (e.g. `+12s density(9v), +8s medium-cong`).

---

## Outputs

### Annotated Video (`output.mp4`)

Each tracked vehicle is annotated with:
- Bounding box coloured by status: **green** (normal), **red** (overspeeding), **purple** (lane crossing)
- Label showing vehicle class and current speed
- Motion trace trail

Lane boundaries are drawn as white outlines labelled L1–L6. The HUD in the top-left corner shows:
- Live and total vehicle counts
- Congestion level and risk score
- Overspeed and lane-crossing counts
- **Adaptive signal status panel** — per-lane phase, countdown, and adjustment reason

Full-width warning banners appear at the bottom for active high-density, overspeed, and lane-crossing alerts.

### Per-frame CSV (`vehicle_speeds.csv`)

| Column | Description |
|---|---|
| `frame` | Frame index |
| `timestamp_sec` | Time in seconds |
| `track_id` | Unique vehicle ID |
| `vehicle_class` | Detected class (Car, Bus, etc.) |
| `speed_kmh` | Estimated speed |
| `overspeed_flag` | 1 if exceeding speed limit |
| `lane_crossing_flag` | 1 if crossing a lane boundary |
| `risk_score` | Composite risk score for the frame |

### Per-vehicle Summary CSV (`vehicle_speed_summary.csv`)

One row per tracked vehicle with `track_id`, `vehicle_class`, `first_frame`, `last_frame`, `max_speed_kmh`, `avg_speed_kmh`, `overspeed_flag`, and `total_lane_crossings`.

### Charts

After processing, three time-series plots are displayed:
- Overspeeding vehicle count over time
- Lane crossing count over time
- Traffic risk score over time

---

## Risk Score Formula

```
risk_score = α × active_vehicles + β × speed_std_dev + γ × congestion_level
```

Where congestion level is 1 (Low), 2 (Medium), or 3 (High), and the default weights are α = 1.2, β = 0.4, γ = 3.0.

---

## Notes

- **GPU strongly recommended.** `yolov8x.pt` at 1280px resolution is slow on CPU. See the Google Colab T4 tip at the top of this file for a free GPU option.
- `LANE_POLYGONS` and `SOURCE` coordinates are specific to a camera angle. Any change in camera position or zoom requires recalibrating both.
- The script detects COCO classes 2 (car), 3 (motorcycle), 5 (bus), and 7 (truck). Other vehicle types are ignored.
- On headless servers (no display), replace `plt.show()` with `plt.savefig("traffic_plots.png")` to save charts to disk instead.
- `adaptive_signal.py` must be in the same directory as `traffic_monitor.py`, as it is imported as a module.
- The adaptive signal controller has no effect on actual physical hardware — it is a simulation layer that annotates the video output. To use it with real signals, integrate `AdaptiveSignalController` with your signal hardware API and call `force_green()` as needed.
