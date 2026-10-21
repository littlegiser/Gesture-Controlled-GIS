from __future__ import annotations

import math
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, jsonify, render_template_string

# ==================== Configuration ====================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

CAMERA_INDEX = 0
HOST = "127.0.0.1"
PORT = 5000

app = Flask(__name__)
lock = threading.Lock()
stop_event = threading.Event()

state = {
    "gesture": "STARTING",
    "stable_gesture": "STARTING",
    "x": 0.5,
    "y": 0.5,
    "zoom_distance": None,
    "sequence": 0,
    "camera_ok": False,
    "message": "Initializing..."
}

# ==================== Web GIS ====================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gesture-Controlled GIS</title>
<link rel="stylesheet"
 href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
* { box-sizing: border-box; }
html,body { margin:0;width:100%;height:100%;font-family:Arial,sans-serif; }
header {
 height:54px;background:#102b49;color:white;padding:0 18px;
 display:flex;align-items:center;justify-content:space-between;
 flex-shrink:0;
}
#layout {
 height:calc(100% - 54px);
 display:flex;
}
#panel {
 width:320px;flex-shrink:0;
 padding:16px;background:#f4f6f9;overflow:auto;
}
#map {
 flex:1;min-width:0;min-height:0;
 background:#e5e7eb;
}
.card {
 background:white;border:1px solid #dce3eb;border-radius:9px;
 padding:14px;margin-bottom:14px;
}
#gesture {font-size:25px;font-weight:bold;color:#17558b;margin:10px 0;}
#message {font-size:13px;color:#667085;overflow-wrap:anywhere;}
.help {font-size:14px;line-height:1.9;color:#344054;}
button {
 padding:9px 12px;border:0;border-radius:6px;
 background:#17558b;color:white;cursor:pointer;margin:5px 3px 0 0;
}
#debug {
 font-family:Consolas,monospace;font-size:12px;
 color:#667085;line-height:1.7;margin-top:8px;white-space:pre-wrap;
}
#tileWarning { display:none; color:#b42318;font-size:12px;margin-top:8px; }
@media(max-width:700px) {
 #layout {flex-direction:column;}
 #panel {width:100%;height:220px;}
 #map {min-height:300px;}
}
</style>
</head>
<body>
<header>
 <strong>Gesture-Controlled GIS</strong>
 <span id="status">Connecting...</span>
</header>
<div id="layout">
 <aside id="panel">
  <div class="card">
   <small>DETECTED GESTURE</small>
   <div id="gesture">STARTING</div>
   <div id="message">Waiting for camera...</div>
  </div>
  <div class="card help">
   <b>✊ Fist (Single)</b> — Pan map<br>
   <b>☝ Index Up (Single)</b> — Zoom In (moving up only)<br>
   <b>🤲 Hands Together (Two)</b> — Zoom Out (moving closer only)<br>
   <br>
   <small>Map remains unchanged when hands leave or release.</small>
  </div>
  <div class="card">
   <button onclick="resetMap()">Reset Map</button>
   <button onclick="toggleFreeze()" id="freezeButton">Freeze</button>
  </div>
  <div class="card">
   <small>DEBUG (Gesture Status)</small>
   <div id="debug">x: -, y: -</div>
   <div id="tileWarning">Network issue, some map tiles failed to load.</div>
  </div>
 </aside>
 <main id="map"></main>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const initialCenter = [19.0760, 72.8777];
const initialZoom = 10;

const map = L.map("map", { zoomControl: true, zoomSnap: 0.1 }).setView(initialCenter, initialZoom);

let tileErrorCount = 0;
const tileLayer = L.tileLayer(
 "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
 {
   maxZoom:19,
   attribution:"&copy; OpenStreetMap contributors"
 }
).addTo(map);

tileLayer.on("tileerror", () => {
  tileErrorCount++;
  if (tileErrorCount >= 15) {
    document.getElementById("tileWarning").style.display = "block";
  }
});

setTimeout(() => map.invalidateSize(), 150);
window.addEventListener("resize", () => map.invalidateSize());

let lastSequence = -1;
let lastPoint = null;
let lastDistance = null;
let lastGesture = "";
let frozen = false;

let smoothPanX = 0;
let smoothPanY = 0;

const gestureEl = document.getElementById("gesture");
const messageEl = document.getElementById("message");
const statusEl = document.getElementById("status");
const freezeButton = document.getElementById("freezeButton");
const debugEl = document.getElementById("debug");

function resetMap() {
 map.setView(initialCenter, initialZoom);
 lastPoint = null;
 lastDistance = null;
}

function setFrozen(value) {
 frozen = value;
 freezeButton.textContent = frozen ? "Unfreeze" : "Freeze";
 statusEl.textContent = frozen ? "Connected · Frozen" : "Connected · Active";
 lastPoint = null;
 lastDistance = null;
}

function toggleFreeze() {
 setFrozen(!frozen);
}

async function update() {
 try {
  const response = await fetch("/state", {cache:"no-store"});
  const s = await response.json();

  statusEl.textContent = s.camera_ok
    ? (frozen ? "Camera connected · Frozen" : "Camera connected · Active")
    : "Camera unavailable";

  gestureEl.textContent = s.stable_gesture;
  messageEl.textContent = s.message;

  if (s.sequence === lastSequence) return;
  lastSequence = s.sequence;

  const g = s.stable_gesture;

  // Reset tracking data on gesture change to prevent map jumping
  if (g !== lastGesture) {
    lastPoint = null;
    lastDistance = null;
    smoothPanX = 0;
    smoothPanY = 0;
  }
  lastGesture = g;

  if (frozen) {
    debugEl.textContent = "FROZEN";
    return;
  }

  const p = [s.x, s.y];
  const size = map.getSize();

  // ----------------- 1. Pan Map (FIST) -----------------
  if (g === "FIST") {
    if (lastPoint !== null) {
      let dx = p[0] - lastPoint[0];
      let dy = p[1] - lastPoint[1];
      
      if (Math.abs(dx) < 0.003) dx = 0;
      if (Math.abs(dy) < 0.003) dy = 0;

      smoothPanX = smoothPanX * 0.6 + dx * 0.4;
      smoothPanY = smoothPanY * 0.6 + dy * 0.4;
      
      map.panBy([-smoothPanX * size.x * 4.0, smoothPanY * size.y * 4.0], {animate:false});
    }
    lastPoint = p;

  // ----------------- 2. Zoom In (POINT) -----------------
  } else if (g === "POINT") {
    if (lastPoint !== null) {
      let dy = p[1] - lastPoint[1];
      // Only zoom in when moving up (dy < -0.005)
      if (dy < -0.005) {
        const dz = Math.min(0.2, -dy * 15.0);
        map.setZoom(Math.min(19, map.getZoom() + dz), {animate:false});
      }
    }
    lastPoint = p;

  // ----------------- 3. Zoom Out (TWO_HANDS) -----------------
  } else if (g === "TWO_HANDS") {
    const d = s.zoom_distance;
    if (typeof d === "number" && lastDistance !== null) {
      const delta = d - lastDistance; // Negative when hands get closer

      // Only zoom out when hands get closer (delta < -0.005)
      if (delta < -0.005) {
        const dz = Math.max(-0.2, delta * 15.0); 
        map.setZoom(Math.max(2, map.getZoom() + dz), {animate:false});
      }
    }
    lastDistance = d;

  // ----------------- Other States -----------------
  } else {
    lastPoint = null;
    lastDistance = null;
  }

  debugEl.textContent =
    "x: " + s.x.toFixed(3) + ", y: " + s.y.toFixed(3) +
    "\nmap zoom: " + map.getZoom().toFixed(2) +
    "\ndist: " + (s.zoom_distance ? s.zoom_distance.toFixed(3) : "none");

 } catch (e) {
  statusEl.textContent = "Cannot connect to Python server";
 }
}

setInterval(update, 33);
</script>
</body>
</html>
"""


# ==================== Model download ====================

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_suffix(".download")

    print("Downloading hand landmark model...")

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "GestureGIS/1.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with open(temp_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if temp_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is too small.")

        temp_path.replace(MODEL_PATH)
        print("Model downloaded:", MODEL_PATH)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Model download failed. Check internet access or download "
            "the official model manually. URL: " + MODEL_URL
        )


# ==================== Gesture recognition ====================

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def classify_single_hand(lm):
    """Classify single hand gestures: FIST or POINT."""
    index_up = finger_up(lm, 8, 6)
    middle_up = finger_up(lm, 12, 10)
    ring_up = finger_up(lm, 16, 14)
    pinky_up = finger_up(lm, 20, 18)

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINT"

    return "OTHER"


CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


def draw_landmarks(frame, lm, color=(70,220,90)):
    h, w = frame.shape[:2]
    pts = [(int(p.x*w), int(p.y*h)) for p in lm]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)

    for x, y in pts:
        cv2.circle(frame, (x,y), 4, (30,80,255), -1)

    cv2.circle(frame, pts[9], 8, (0,255,0), -1)
    cv2.circle(frame, pts[8], 8, (0,255,255), -1)


# ==================== Camera worker ====================

def camera_worker():
    cap = None
    landmarker = None

    try:
        ensure_model()

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,  # Enable two-hand detection
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.50
        )

        landmarker = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        candidate = "NO HAND"
        candidate_count = 0
        stable = "NO HAND"

        smooth_x = 0.5
        smooth_y = 0.5
        alpha = 0.35

        with lock:
            state["camera_ok"] = True
            state["message"] = "Camera and MediaPipe are ready."

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = "NO HAND"
            x, y = 0.5, 0.5
            zoom_distance = None

            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)

                if num_hands >= 2:
                    # Two-hand mode: Zoom Out
                    lm1 = result.hand_landmarks[0]
                    lm2 = result.hand_landmarks[1]
                    p1 = lm1[9]
                    p2 = lm2[9]
                    dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                    
                    gesture = "TWO_HANDS"
                    zoom_distance = dist
                    
                    # Use the midpoint of both hands to prevent frontend errors
                    smooth_x = alpha * ((p1.x + p2.x) / 2) + (1-alpha) * smooth_x
                    smooth_y = alpha * ((p1.y + p2.y) / 2) + (1-alpha) * smooth_y
                    x, y = smooth_x, smooth_y

                    draw_landmarks(frame, lm1, (0, 0, 255))
                    draw_landmarks(frame, lm2, (0, 0, 255))
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(frame, pt1, pt2, (255, 200, 0), 3)
                    cv2.putText(frame, f"DIST: {dist:.2f}", (16, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,150,0), 2)
                    
                elif num_hands == 1:
                    lm = result.hand_landmarks[0]
                    raw_gesture = classify_single_hand(lm)
                    
                    if raw_gesture == "FIST":
                        p = lm[9]
                        gesture = "FIST"
                    elif raw_gesture == "POINT":
                        p = lm[8]
                        gesture = "POINT"
                    else:
                        p = lm[9]
                        gesture = "OTHER"

                    smooth_x = alpha * p.x + (1-alpha) * smooth_x
                    smooth_y = alpha * p.y + (1-alpha) * smooth_y
                    x, y = smooth_x, smooth_y

                    color = (70, 220, 90) if gesture != "OTHER" else (150, 150, 150)
                    draw_landmarks(frame, lm, color)
                    
                    if gesture in ["FIST", "POINT"]:
                        tx = max(10, min(w - 150, int(p.x * w) - 30))
                        ty = max(40, min(h - 20, int(p.y * h) - 20))
                        cv2.putText(frame, gesture, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if gesture == candidate:
                candidate_count += 1
            else:
                candidate = gesture
                candidate_count = 1

            if candidate_count >= 3:
                stable = candidate

            with lock:
                state["gesture"] = gesture
                state["stable_gesture"] = stable
                state["x"] = float(x)
                state["y"] = float(y)
                state["zoom_distance"] = (
                    float(zoom_distance)
                    if zoom_distance is not None else None
                )
                state["camera_ok"] = True
                state["message"] = f"Raw: {gesture} | Stable: {stable}"
                state["sequence"] += 1

            cv2.putText(frame, f"Gesture: {gesture}", (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Stable: {stable}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Gesture GIS - Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_event.set()
                break

    except Exception as exc:
        print("\n[Camera worker error]", repr(exc))
        with lock:
            state["camera_ok"] = False
            state["gesture"] = "ERROR"
            state["stable_gesture"] = "ERROR"
            state["message"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


# ==================== Flask routes ====================

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/state")
def get_state():
    with lock:
        return jsonify(dict(state))

@app.route("/health")
def health():
    return jsonify({"ok": True, "api": "MediaPipe Tasks"})


def main():
    worker = threading.Thread(
        target=camera_worker,
        name="camera-worker",
        daemon=True
    )
    worker.start()

    url = f"http://{HOST}:{PORT}"
    print("\nGesture-Controlled Web GIS (Custom Multi-Hand)")
    print(f"Open {url} in Chrome or Edge.")
    
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()