# Traffic Monitor & Speed Estimation System

This repository contains an end-to-end computer vision pipeline that performs vehicle detection, tracking, real-world speed estimation, and traffic density analysis using a monocular traffic video camera feed. 

The system leverages **YOLOv8** for high-accuracy detection, **ByteTrack** for persistent vehicle tracking, and the **Supervision** library for advanced frame annotation and zone filtering. Perspective transformation is utilized to translate pixels into physical ground coordinates for realistic speed tracking.

---

## Key Features

* **Object Detection & Classification:** Detects and distinguishes between Cars, Motorcycles, Buses, and Trucks (`yolov8x`).
* **Perspective Speed Estimation:** Uses a Homography/Perspective Transform matrix to accurately compute real-world vehicle velocities (km/h) across a calibrated ground plane.
* **Live Alerts & Annotations:** Dynamic HUD displaying live vehicle count, total unique vehicle counters, and instant visual flag overlays (Banners) for overspeeding or high-density traffic.
* **Comprehensive Data Logging:** Automatically outputs two CSV logs:
  1. `vehicle_speeds.csv`: Frame-by-frame instantaneous tracking, velocity data, and timestamps.
  2. `vehicle_speed_summary.csv`: Aggregated session metrics per vehicle (Max speed, Avg speed, Active frames, Overspeed flags).

---

## Recommended Execution Environment

### Google Colab (Highly Recommended)
Because this pipeline processes object detection with a high-parameter footprint (`yolov8x.pt`) at a native resolution of `1280px`, execution on a local CPU will be slow. 

It is highly recommended to run this code using **Google Colab** with a **T4 GPU** runtime. 
* **Why T4 GPU?** The T4 provides optimal Tensor Core acceleration for mixed-precision inference with Ultralytics, enabling significantly faster frame-per-second (FPS) processing rates during target video rendering.
* **To activate in Colab:** Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU` -> Click `Save`.

---

## Installation & Setup

If running locally or within a notebook environment, install the required dependencies:

```bash
pip install opencv-python numpy supervision ultralytics tqdm
