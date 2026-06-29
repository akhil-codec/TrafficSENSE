
import cv2
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np



#  Tunable parameters
BASE_GREEN_SEC   = 20.0   # default green duration (seconds)
BASE_RED_SEC     = 15.0   # default red duration (seconds)
MIN_GREEN_SEC    = 8.0    # never go below this
MAX_GREEN_SEC    = 60.0   # cap to avoid starvation on other lanes
YELLOW_SEC       = 3.0    # fixed yellow phase

# How strongly each factor pulls the green time
DENSITY_WEIGHT   = 1.5    # extra seconds per vehicle above threshold
SPEED_VAR_WEIGHT = 0.4    # extra seconds per unit of speed std-dev
CONGESTION_BONUS = {"low": 0, "medium": 8, "high": 20}   # flat bonus (sec)

# Density thresholds (vehicles in lane)
LOW_DENSITY_THRESH    = 3
MEDIUM_DENSITY_THRESH = 7



#  Data structures
@dataclass
class LaneStats:
    """Per-lane snapshot passed in each frame."""
    lane_id: int
    vehicle_count: int          = 0
    avg_speed_kmh: float        = 0.0
    speed_std_kmh: float        = 0.0
    congestion_level: str       = "low"   # "low" | "medium" | "high"
    lane_crossing_count: int    = 0
    overspeed_count: int        = 0


@dataclass
class SignalPhase:
    """Represents one traffic light phase."""
    color: str          # "GREEN" | "YELLOW" | "RED"
    duration_sec: float
    elapsed_sec: float  = 0.0

    @property
    def remaining_sec(self) -> float:
        return max(0.0, self.duration_sec - self.elapsed_sec)

    @property
    def progress(self) -> float:
        """0.0 → 1.0 fraction of phase completed."""
        if self.duration_sec == 0:
            return 1.0
        return min(1.0, self.elapsed_sec / self.duration_sec)


@dataclass
class LaneSignalState:
    """Full signal state for one lane."""
    lane_id: int
    phase: SignalPhase = field(default_factory=lambda: SignalPhase("RED", BASE_RED_SEC))
    computed_green_sec: float = BASE_GREEN_SEC
    reason: str = ""            # human-readable explanation of last adjustment

    # Stats history (rolling)
    _density_history: List[int]   = field(default_factory=list)
    _speed_var_history: List[float] = field(default_factory=list)

    def update_history(self, stats: LaneStats, window: int = 30) -> None:
        self._density_history.append(stats.vehicle_count)
        self._speed_var_history.append(stats.speed_std_kmh)
        if len(self._density_history) > window:
            self._density_history.pop(0)
            self._speed_var_history.pop(0)

    @property
    def avg_density(self) -> float:
        return float(np.mean(self._density_history)) if self._density_history else 0.0

    @property
    def avg_speed_var(self) -> float:
        return float(np.mean(self._speed_var_history)) if self._speed_var_history else 0.0



#  Controller
class AdaptiveSignalController:

    def __init__(self, num_lanes: int, fps: float, cycle_offset_sec: float = 10.0):

        self.fps = fps
        self.num_lanes = num_lanes
        self.states: Dict[int, LaneSignalState] = {}

        for lane_id in range(num_lanes):
            state = LaneSignalState(lane_id=lane_id)
            # Stagger starting phase so lanes alternate naturally
            offset = lane_id * cycle_offset_sec
            if offset < BASE_GREEN_SEC:
                state.phase = SignalPhase("GREEN", BASE_GREEN_SEC, elapsed_sec=offset)
            else:
                state.phase = SignalPhase("RED",   BASE_RED_SEC,
                                          elapsed_sec=offset - BASE_GREEN_SEC)
            self.states[lane_id] = state

        self._last_frame: int = 0
        self._global_override: Optional[str] = None   # e.g. "EMERGENCY"

    # ── Public API 

    def update(self, frame_idx: int, lane_stats: List[LaneStats]) -> None:

        dt = (frame_idx - self._last_frame) / self.fps   # seconds since last call
        self._last_frame = frame_idx

        stats_by_id = {s.lane_id: s for s in lane_stats}

        for lane_id, state in self.states.items():
            stats = stats_by_id.get(lane_id, LaneStats(lane_id=lane_id))
            state.update_history(stats)

            # Recompute desired green time for this lane
            state.computed_green_sec = self._compute_green(stats, state)

            # Advance phase clock
            state.phase.elapsed_sec += dt

            # Transition phases when time is up
            if state.phase.elapsed_sec >= state.phase.duration_sec:
                self._advance_phase(state)

    def get_display_status(self) -> Dict[int, dict]:

        out = {}
        for lane_id, state in self.states.items():
            phase = state.phase
            color_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(phase.color, "⚪")
            out[lane_id] = {
                "color":     phase.color,
                "remaining": round(phase.remaining_sec, 1),
                "label":     f"{color_emoji} L{lane_id+1} {phase.color} — {phase.remaining_sec:.0f}s left",
                "reason":    state.reason,
                "progress":  phase.progress,
                "computed_green": round(state.computed_green_sec, 1),
            }
        return out

    def get_active_green_lanes(self) -> List[int]:
        """Returns list of lane_ids currently in GREEN phase."""
        return [lid for lid, s in self.states.items() if s.phase.color == "GREEN"]

    def force_green(self, lane_id: int, duration_sec: Optional[float] = None) -> None:
        """Emergency override — immediately give a lane a GREEN phase."""
        if lane_id in self.states:
            dur = duration_sec or self.states[lane_id].computed_green_sec
            self.states[lane_id].phase = SignalPhase("GREEN", dur)
            self.states[lane_id].reason = "FORCED GREEN (override)"

    # ── Internal helpers 
    

    def _compute_green(self, stats: LaneStats, state: LaneSignalState) -> float:

        base = BASE_GREEN_SEC
        reasons = []

        # Density component
        density_excess = max(0, stats.vehicle_count - LOW_DENSITY_THRESH)
        density_add = DENSITY_WEIGHT * density_excess
        if density_add > 0:
            reasons.append(f"+{density_add:.0f}s density({stats.vehicle_count}v)")

        # Speed variation component (chaotic traffic → give more time to clear)
        var_add = SPEED_VAR_WEIGHT * stats.speed_std_kmh
        if var_add > 1:
            reasons.append(f"+{var_add:.0f}s spd-var")

        # Congestion level flat bonus
        cong_add = CONGESTION_BONUS.get(stats.congestion_level, 0)
        if cong_add > 0:
            reasons.append(f"+{cong_add}s {stats.congestion_level}-cong")

        green = base + density_add + var_add + cong_add
        green = float(np.clip(green, MIN_GREEN_SEC, MAX_GREEN_SEC))

        state.reason = ", ".join(reasons) if reasons else "nominal"
        return green

    def _advance_phase(self, state: LaneSignalState) -> None:
        """Cycle GREEN → YELLOW → RED → GREEN with adaptive durations."""
        current = state.phase.color
        overflow = state.phase.elapsed_sec - state.phase.duration_sec  # carry leftover

        if current == "GREEN":
            state.phase = SignalPhase("YELLOW", YELLOW_SEC, elapsed_sec=overflow)
        elif current == "YELLOW":
            state.phase = SignalPhase("RED",    BASE_RED_SEC, elapsed_sec=overflow)
        elif current == "RED":
            # Use the freshly computed adaptive green duration
            state.phase = SignalPhase("GREEN", state.computed_green_sec, elapsed_sec=overflow)



#  OpenCV overlay helper (optional)

SIGNAL_COLOR_BGR = {
    "GREEN":  (0,   200,  0),
    "YELLOW": (0,   200, 200),
    "RED":    (0,   0,   220),
}

def draw_signal_hud(frame, signal_status: dict, x: int = 0, y: int = 235) -> None:
    

    n = len(signal_status)
    panel_w, row_h = 310, 42
    panel_h = n * row_h + 28
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    # Header
    cv2.putText(frame, "ADAPTIVE SIGNAL STATUS", (x + 8, y + 17),
                font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(frame, (x + 8, y + 22), (x + panel_w - 8, y + 22), (80, 80, 80), 1)

    for i, (lane_id, info) in enumerate(sorted(signal_status.items())):
        row_y = y + 28 + i * row_h
        color_bgr = SIGNAL_COLOR_BGR.get(info["color"], (150, 150, 150))

        # Colored circle indicator
        cv2.circle(frame, (x + 16, row_y + 14), 8, color_bgr, -1, cv2.LINE_AA)

        # Lane label + countdown
        main_text = f"Lane {lane_id+1}: {info['color']}  {info['remaining']:.0f}s"
        cv2.putText(frame, main_text, (x + 30, row_y + 18),
                    font, 0.46, color_bgr, 1, cv2.LINE_AA)

        # Progress bar
        bar_x1, bar_y  = x + 30, row_y + 25
        bar_w, bar_h   = panel_w - 45, 5
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        filled = int(bar_w * (1.0 - info["progress"]))   # shrinks as phase progresses
        if filled > 0:
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled, bar_y + bar_h),
                          color_bgr, -1)

        # Reason (small grey text)
        if info["reason"] and info["reason"] != "nominal":
            reason_short = info["reason"][:42]   # truncate for HUD width
            cv2.putText(frame, reason_short, (x + 30, row_y + 38),
                        font, 0.30, (130, 130, 130), 1, cv2.LINE_AA)


