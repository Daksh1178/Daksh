# Hand Gesture Cursor Control (Desktop App)

A Python desktop application that lets you control your mouse cursor using real-time hand gestures from a webcam.

## Features

- Real-time hand detection and tracking with **MediaPipe**
- Camera feed with hand-landmark overlay via **OpenCV**
- Mouse control via **PyAutoGUI**
- Gesture mapping:
  - **Move cursor** → Index finger movement
  - **Left click** → Thumb + index finger quick pinch
  - **Right click** → Thumb + middle finger pinch
  - **Double click** → Two quick thumb + index pinches
  - **Drag and drop** → Thumb + index pinch-and-hold, release to drop
  - **Scroll** → Move two raised fingers (index + middle) vertically
- Jitter reduction via smoothing/interpolation
- On-screen gesture/status display
- Toggle control mode ON/OFF
- Sensitivity slider (OpenCV trackbar)
- FPS counter

## Project Structure

- `cursor_gesture_control.py` - Main application code
- `requirements.txt` - Python dependencies

## Installation

1. Make sure Python 3.9+ is installed.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python cursor_gesture_control.py
```

## Runtime Controls

- `c` → Toggle cursor control ON/OFF
- `q` → Quit the app

## Gesture Tips

- Keep your hand in clear view under normal indoor lighting.
- For accurate pinch gestures, keep your thumb and target finger clearly separated between actions.
- For drag-and-drop: hold thumb+index pinch for ~0.35s to start dragging, then release to drop.

## Notes

- If your operating system blocks mouse control, grant accessibility/input permissions to Python/terminal.
- Webcam access permission may also be required.
