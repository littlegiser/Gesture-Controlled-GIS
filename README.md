# Gesture-Controlled Web GIS (VS Code)

A local learning prototype using Python, OpenCV, MediaPipe, Flask, and Leaflet.

## 1. Recommended environment

- Windows 10/11
- Python 3.11 (64-bit recommended)
- VS Code with the Python extension
- A webcam
- Internet connection for OpenStreetMap tiles

Python 3.11 is recommended because computer-vision packages may not yet support every newest Python version.

## 2. Open the project

Extract the ZIP file, then open the `gesture_gis_vscode` folder in VS Code.

Open **Terminal → New Terminal** and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable, install Python 3.11 64-bit first, or use `python -m venv .venv` with a compatible Python version.

## 3. Run

```powershell
python app.py
```

The terminal should print:

```text
Open http://127.0.0.1:5000 in your browser.
```

Open that address in Chrome or Edge. A separate OpenCV window should show the camera feed and hand landmarks.

## 4. Gestures

- **POINT**: index finger up, other three fingers curled. Move the finger to pan the map.
- **PINCH / ZOOM**: index finger up, other three fingers curled, with thumb visible. Change the thumb-index distance to zoom.
- **OPEN_PALM**: four fingers up. Reset the map to the initial view.
- **FIST**: four fingers curled. Toggle freeze/unfreeze on the transition into a fist.
- **Q** or **Esc** in the camera window: stop the camera worker.

The gesture classifier uses geometric rules and temporal debounce, not a custom-trained model. Hand pose, camera angle, lighting, and whether the thumb is visible can affect recognition.

## 5. Troubleshooting

### Camera does not open
1. Close Zoom/Teams/other applications using the webcam.
2. Check Windows **Settings → Privacy & security → Camera** and allow desktop apps.
3. In `app.py`, change `cv2.VideoCapture(0, cv2.CAP_DSHOW)` to camera index `1` if you have multiple cameras.

### MediaPipe installation fails
Use 64-bit Python 3.11 and recreate the virtual environment. Avoid mixing packages from a global Python installation with this `.venv`.

### Map is blank
The Leaflet library and OpenStreetMap tiles load from the internet. Check the network, proxy, or firewall. Camera processing itself is local.

### Hand gesture is misclassified
Use a plain background, keep the whole hand in view, face the palm toward the camera, and avoid very low light. You can adjust the five-frame debounce in `app.py`.

## Notes

- This is a prototype, not a production-grade gesture interface.
- The Flask server binds only to `127.0.0.1`, so it is intended for local use.
- OpenStreetMap tile usage is subject to the OpenStreetMap tile usage policy.
