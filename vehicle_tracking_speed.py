import cv2
import csv
import time
import math
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

 
VIDEO_PATH              = "traffic_surveillance_video_data.mp4"
OUTPUT_VIDEO            = "output_tracking_speed.mp4"
CSV_PER_FRAME           = "vehicle_speeds.csv"
CSV_SUMMARY             = "vehicle_speed_summary.csv"
MODEL_PATH              = "yolov8n.pt"          

#  Speed estimation calibration 
PIXELS_PER_METER        = 8.0       # px/m — CALIBRATE for your video (see above)
SPEED_LIMIT_KMH         = 60.0      # km/h overspeed threshold
SPEED_SMOOTH_FRAMES     = 7         # moving-average window for speed smoothing

#  YOLO 
CONFIDENCE_THRESH       = 0.40

#  DeepSORT 
DEEPSORT_MAX_AGE        = 30        # frames to keep a lost track alive
DEEPSORT_N_INIT         = 3         # consecutive detections to confirm a track
DEEPSORT_MAX_COSINE     = 0.4       # re-ID similarity threshold

#  Display 
TRAIL_LENGTH            = 40        # centroid trail history length in frames
SHOW_WINDOW             = True      

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Colours  {track_id mod N}  — enough variety for many vehicles
PALETTE = [
    (255, 128,   0), (  0, 200, 255), (180,   0, 255), ( 50, 255,  50),
    (255,  50, 150), (  0, 255, 180), (255, 200,   0), ( 50, 100, 255),
    (255,  80,  80), (  0, 180, 128), (200, 200,   0), (128,   0, 200),
]
OVERSPEED_COLOR = (0, 0, 255)       # red for overspeeding vehicles
NORMAL_COLOR    = (0, 230, 0)       # green for normal-speed vehicles



def pixel_speed_to_kmh(prev_c, curr_c, fps: float, px_per_m: float) -> float:
    """Convert frame-to-frame centroid displacement → km/h."""
    dx = curr_c[0] - prev_c[0]
    dy = curr_c[1] - prev_c[1]
    px_dist    = math.hypot(dx, dy)
    speed_mps  = px_dist * fps / px_per_m
    return speed_mps * 3.6


def smooth_speed(speed_history: deque) -> float:
    """Exponentially weighted moving average (recent frames weighted more)."""
    vals = list(speed_history)
    if not vals:
        return 0.0
    weights = [math.exp(0.3 * i) for i in range(len(vals))]
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def track_color(track_id, overspeeding: bool) -> tuple:
    # DeepSORT may return track_id as a string — cast to int before modulo
    return OVERSPEED_COLOR if overspeeding else PALETTE[int(track_id) % len(PALETTE)]


def draw_semi_transparent_box(frame, x1, y1, x2, y2,
                               color=(10, 10, 10), alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_speedometer_arc(frame, cx, cy, radius, speed, limit):
    """Draw a small arc gauge around vehicle centroid showing speed ratio."""
    ratio      = min(speed / max(limit, 1), 1.5)        # cap at 150 %
    end_angle  = int(-90 + ratio * 270)                 # from top, clockwise
    color      = OVERSPEED_COLOR if speed > limit else (0, 220, 100)
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                0, -90, end_angle, color, 2)


def draw_warning_banner(frame, text: str, vh: int, vw: int):
    banner_y = vh - 55
    draw_semi_transparent_box(frame, 0, banner_y, vw, vh, (0, 0, 160), 0.60)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.82, 2)
    tx, ty = (vw - tw) // 2, banner_y + 36
    cv2.putText(frame, text, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 0, 0), 2)       # shadow
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2)


def draw_vehicle_label(frame, x1, y1, track_id, label, speed, overspeeding):
    """Draw a multi-line ID + speed label above the bounding box."""
    color    = OVERSPEED_COLOR if overspeeding else PALETTE[int(track_id) % len(PALETTE)]
    tag      = f"ID:{track_id}  {label}"
    spd_txt  = f"{speed:.1f} km/h"
    flag_txt = " [OVERSPEED]" if overspeeding else ""

    for i, (txt, clr) in enumerate([
        (tag,               (240, 240, 240)),
        (spd_txt + flag_txt, color),
    ]):
        offset_y = y1 - 28 + i * 18
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame,
                      (x1 - 1, offset_y - th - 2),
                      (x1 + tw + 3, offset_y + 2),
                      (20, 20, 20), -1)
        cv2.putText(frame, txt, (x1 + 1, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1)



def main():
    #  Validate input 
    if not Path(VIDEO_PATH).exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        print("       Place your traffic video in the same folder and update VIDEO_PATH.")
        return

    #  Load YOLO 
    print("[INFO] Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)

    #  Load DeepSORT 
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nms_max_overlap=1.0,
        max_cosine_distance=DEEPSORT_MAX_COSINE,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=False,
    )
    print("[INFO] DeepSORT tracker ready.")

    #  Open video 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    VW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {VW}×{VH} @ {FPS:.1f} fps  |  {TOTAL} frames")
    print(f"[INFO] Speed limit: {SPEED_LIMIT_KMH} km/h  |  px/m: {PIXELS_PER_METER}\n")

    #  Video writer 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VW, VH))

    #  Per-frame CSV 
    f_frame   = open(CSV_PER_FRAME, "w", newline="")
    w_frame   = csv.writer(f_frame)
    w_frame.writerow([
        "frame", "timestamp_sec", "track_id", "vehicle_class",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "centroid_x", "centroid_y",
        "instant_speed_kmh", "smooth_speed_kmh",
        "overspeed_flag"
    ])

    #  Per-track state 
    track_centroids  = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    track_speed_hist = defaultdict(lambda: deque(maxlen=SPEED_SMOOTH_FRAMES))
    track_class      = {}          # track_id → class label string
    track_max_speed  = defaultdict(float)
    track_all_speeds = defaultdict(list)
    track_first_frame= {}
    track_last_frame = {}

    #  Processing loop 
    frame_num          = 0
    t_start            = time.time()
    overspeeding_now   = set()     # track IDs overspeeding this frame

    print("[INFO] Processing … press Q in the window to quit early.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        ts = frame_num / FPS

        #  YOLO detect 
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESH)[0]

        detections = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            # DeepSORT expects [left, top, width, height], confidence, class_label
            detections.append(([x1, y1, w, h], conf, VEHICLE_CLASSES[cls]))

        # ── DeepSORT update 
        tracks = tracker.update_tracks(detections, frame=frame)

        overspeeding_now = set()

        for trk in tracks:
            if not trk.is_confirmed():
                continue

            tid  = int(trk.track_id)   # cast: DeepSORT may yield a string
            ltrb = trk.to_ltrb()
            x1, y1, x2, y2 = (
                max(0, int(ltrb[0])),
                max(0, int(ltrb[1])),
                min(VW, int(ltrb[2])),
                min(VH, int(ltrb[3])),
            )
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Class label (keep first assignment)
            if tid not in track_class:
                det_cls = getattr(trk, "det_class", None)
                track_class[tid] = det_cls if det_cls else "Vehicle"

            # First / last frame bookkeeping
            track_first_frame.setdefault(tid, frame_num)
            track_last_frame[tid] = frame_num

            #  Speed estimation 
            centroids = track_centroids[tid]
            centroids.append((cx, cy))

            instant_speed = 0.0
            if len(centroids) >= 2:
                instant_speed = pixel_speed_to_kmh(
                    centroids[-2], centroids[-1], FPS, PIXELS_PER_METER
                )

            track_speed_hist[tid].append(instant_speed)
            avg_speed = smooth_speed(track_speed_hist[tid])

            track_all_speeds[tid].append(avg_speed)
            if avg_speed > track_max_speed[tid]:
                track_max_speed[tid] = avg_speed

            is_over = avg_speed > SPEED_LIMIT_KMH
            if is_over:
                overspeeding_now.add(tid)

            #  Draw vehicle 
            color = track_color(tid, is_over)
            thickness = 3 if is_over else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Centroid dot
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Trail
            pts = list(centroids)
            for i in range(1, len(pts)):
                alpha_factor = i / len(pts)
                tc = tuple(int(c * alpha_factor) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], tc, 2)

            # Speed gauge arc
            draw_speedometer_arc(frame, cx, cy, 22, avg_speed, SPEED_LIMIT_KMH)

            # Label
            draw_vehicle_label(frame, x1, y1,
                                tid, track_class[tid], avg_speed, is_over)

            # Overspeed flash box
            if is_over:
                flash_overlay = frame.copy()
                cv2.rectangle(flash_overlay, (x1, y1), (x2, y2),
                              OVERSPEED_COLOR, -1)
                cv2.addWeighted(flash_overlay, 0.12, frame, 0.88, 0, frame)

            #  Write per-frame CSV row 
            w_frame.writerow([
                frame_num, round(ts, 3), tid, track_class[tid],
                x1, y1, x2, y2, cx, cy,
                round(instant_speed, 2), round(avg_speed, 2),
                int(is_over)
            ])

        #  Stats panel 
        active_tracks  = sum(1 for t in tracks if t.is_confirmed())
        n_over         = len(overspeeding_now)

        draw_semi_transparent_box(frame, 0, 0, 300, 195)

        cv2.putText(frame, "TRAFFIC MONITOR",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.line(frame, (8, 26), (292, 26), (100, 100, 100), 1)

        cv2.putText(frame, f"Frame         : {frame_num:>5d} / {TOTAL}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(frame, f"Time          : {ts:>6.1f} s",
                    (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(frame, f"Tracked Vehicles : {active_tracks}",
                    (8, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (100, 220, 255), 2)
        cv2.putText(frame, f"Speed Limit   : {SPEED_LIMIT_KMH:.0f} km/h",
                    (8, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 1)
        cv2.putText(frame,
                    f"Overspeeding  : {n_over}",
                    (8, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                    OVERSPEED_COLOR if n_over > 0 else (180, 180, 180), 2)

        # List overspeeding IDs
        if overspeeding_now:
            id_txt = "IDs: " + ", ".join(str(i) for i in sorted(overspeeding_now)[:8])
            cv2.putText(frame, id_txt,
                        (8, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.46, OVERSPEED_COLOR, 1)

        # Unique vehicles seen
        cv2.putText(frame, f"Total Unique IDs : {len(track_class)}",
                    (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # Legend
        cv2.circle(frame,  (10, 188), 5, NORMAL_COLOR, -1)
        cv2.putText(frame, "Normal",    (18, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.40, NORMAL_COLOR, 1)
        cv2.circle(frame,  (80, 188), 5, OVERSPEED_COLOR, -1)
        cv2.putText(frame, "Overspeed", (88, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.40, OVERSPEED_COLOR, 1)

        #  Global overspeed warning banner 
        if n_over > 0:
            draw_warning_banner(
                frame,
                f"!! OVERSPEED ALERT — {n_over} VEHICLE(S) EXCEEDING {SPEED_LIMIT_KMH:.0f} km/h !!",
                VH, VW
            )

        #  Processing FPS 
        proc_fps = frame_num / max(time.time() - t_start, 0.001)
        cv2.putText(frame, f"Proc FPS: {proc_fps:.1f}",
                    (VW - 160, VH - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("Vehicle Tracking & Speed Detection [Q to quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Early exit by user.")
                break

        if frame_num % 30 == 0:
            print(f"  Frame {frame_num:>5d}/{TOTAL}  |  Tracked={active_tracks}  "
                  f"|  Overspeeding={n_over}  |  Proc-FPS={proc_fps:.1f}")

    #  WRITE SUMMARY CSV

    cap.release()
    writer.release()
    f_frame.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n[INFO] Writing speed summary …")
    with open(CSV_SUMMARY, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "track_id", "vehicle_class",
            "first_frame", "last_frame", "duration_sec",
            "max_speed_kmh", "avg_speed_kmh", "min_speed_kmh",
            "overspeed_flag", "overspeed_note"
        ])
        for tid in sorted(track_all_speeds.keys()):
            speeds  = track_all_speeds[tid]
            if not speeds:
                continue
            dur_sec  = (track_last_frame[tid] - track_first_frame[tid]) / FPS
            max_spd  = max(speeds)
            avg_spd  = sum(speeds) / len(speeds)
            min_spd  = min(speeds)
            is_over  = max_spd > SPEED_LIMIT_KMH
            note     = f"Exceeded limit by {max_spd - SPEED_LIMIT_KMH:.1f} km/h" \
                       if is_over else "Within limit"
            sw.writerow([
                tid, track_class.get(tid, "Unknown"),
                track_first_frame.get(tid, ""), track_last_frame.get(tid, ""),
                round(dur_sec, 2),
                round(max_spd, 2), round(avg_spd, 2), round(min_spd, 2),
                int(is_over), note
            ])

    #  Final report 
    all_overspeed = [tid for tid, spd in track_max_speed.items()
                     if spd > SPEED_LIMIT_KMH]

    print("\n" + "═" * 65)
    print(f"  Output video      : {OUTPUT_VIDEO}")
    print(f"  Per-frame CSV     : {CSV_PER_FRAME}")
    print(f"  Summary CSV       : {CSV_SUMMARY}")
    print(f"  Total frames      : {frame_num}")
    print(f"  Unique vehicle IDs: {len(track_class)}")
    print(f"  Overspeeding IDs  : {len(all_overspeed)}  →  {sorted(all_overspeed)}")
    if track_max_speed:
        top_id  = max(track_max_speed, key=track_max_speed.get)
        top_spd = track_max_speed[top_id]
        print(f"  Fastest vehicle   : ID {top_id}  "
              f"({track_class.get(top_id,'?')})  @  {top_spd:.1f} km/h")
    print("═" * 65)


if __name__ == "__main__":
    main()
