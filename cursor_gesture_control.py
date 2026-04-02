import math
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui


class GestureMouseController:
    """Control system cursor using real-time hand gestures from webcam feed."""

    def __init__(self):
        # MediaPipe hand model setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen / cursor state
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_points = deque(maxlen=6)

        # Control state
        self.control_enabled = True
        self.last_gesture = "Idle"

        # Click / drag states
        self.index_thumb_pinch_active = False
        self.pinch_start_time = 0.0
        self.dragging = False
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.last_scroll_time = 0.0

        # Gesture timing thresholds
        self.drag_hold_seconds = 0.35
        self.double_click_window = 0.35
        self.action_cooldown = 0.25

        # Scroll state
        self.prev_scroll_y = None

        # FPS tracking
        self.prev_frame_time = time.time()

    @staticmethod
    def _landmark_xy(landmarks, idx, frame_w, frame_h):
        lm = landmarks[idx]
        return int(lm.x * frame_w), int(lm.y * frame_h)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _fingers_up(self, hand_landmarks):
        """Simple finger-up heuristic for non-rotated palm scenarios."""
        lm = hand_landmarks.landmark
        fingers = {
            "index": lm[8].y < lm[6].y,
            "middle": lm[12].y < lm[10].y,
            "ring": lm[16].y < lm[14].y,
            "pinky": lm[20].y < lm[18].y,
        }
        return fingers

    def _smooth_cursor(self, x, y):
        self.smooth_points.append((x, y))
        avg_x = int(sum(p[0] for p in self.smooth_points) / len(self.smooth_points))
        avg_y = int(sum(p[1] for p in self.smooth_points) / len(self.smooth_points))
        return avg_x, avg_y

    def _to_screen_coords(self, x, y, frame_w, frame_h, margin=60):
        # Map hand movement inside a margin-box to full screen for better control range.
        x = max(margin, min(frame_w - margin, x))
        y = max(margin, min(frame_h - margin, y))

        x_norm = (x - margin) / (frame_w - 2 * margin)
        y_norm = (y - margin) / (frame_h - 2 * margin)

        x_screen = int((1 - x_norm) * self.screen_w)  # mirror X to match selfie camera
        y_screen = int(y_norm * self.screen_h)
        return x_screen, y_screen

    def _process_hand(self, frame, hand_landmarks, scroll_sensitivity):
        frame_h, frame_w, _ = frame.shape
        lm = hand_landmarks.landmark

        # Key landmarks
        thumb_tip = self._landmark_xy(lm, 4, frame_w, frame_h)
        index_tip = self._landmark_xy(lm, 8, frame_w, frame_h)
        middle_tip = self._landmark_xy(lm, 12, frame_w, frame_h)
        wrist = self._landmark_xy(lm, 0, frame_w, frame_h)
        middle_mcp = self._landmark_xy(lm, 9, frame_w, frame_h)

        # Dynamic threshold based on hand size (more robust across distances)
        hand_scale = max(35, self._distance(wrist, middle_mcp))
        pinch_threshold = hand_scale * 0.45

        dist_index_thumb = self._distance(index_tip, thumb_tip)
        dist_middle_thumb = self._distance(middle_tip, thumb_tip)

        now = time.time()
        fingers = self._fingers_up(hand_landmarks)

        # Cursor movement (index finger driven)
        if self.control_enabled:
            sx, sy = self._to_screen_coords(index_tip[0], index_tip[1], frame_w, frame_h)
            sx, sy = self._smooth_cursor(sx, sy)
            pyautogui.moveTo(sx, sy, _pause=False)
            self.last_gesture = "Move Cursor"

        # Right click: thumb + middle pinch
        right_pinch = dist_middle_thumb < pinch_threshold
        if (
            self.control_enabled
            and right_pinch
            and dist_index_thumb >= pinch_threshold * 0.9
            and not self.dragging
            and now - self.last_right_click_time > self.action_cooldown
        ):
            pyautogui.rightClick()
            self.last_right_click_time = now
            self.last_gesture = "Right Click"

        # Left click / double-click / drag via thumb + index pinch
        index_pinch = dist_index_thumb < pinch_threshold

        if index_pinch and not self.index_thumb_pinch_active:
            self.index_thumb_pinch_active = True
            self.pinch_start_time = now

        if index_pinch and self.index_thumb_pinch_active and not self.dragging:
            if now - self.pinch_start_time >= self.drag_hold_seconds:
                if self.control_enabled:
                    pyautogui.mouseDown()
                    self.dragging = True
                    self.last_gesture = "Drag (Hold)"

        if not index_pinch and self.index_thumb_pinch_active:
            pinch_duration = now - self.pinch_start_time
            self.index_thumb_pinch_active = False

            if self.dragging:
                if self.control_enabled:
                    pyautogui.mouseUp()
                self.dragging = False
                self.last_gesture = "Drop"

            elif pinch_duration < self.drag_hold_seconds and now - self.last_left_click_time > 0.12:
                # Quick pinch interpreted as click; two quick pinches -> double click
                if now - self.last_left_click_time <= self.double_click_window:
                    if self.control_enabled:
                        pyautogui.doubleClick()
                    self.last_gesture = "Double Click"
                else:
                    if self.control_enabled:
                        pyautogui.click()
                    self.last_gesture = "Left Click"
                self.last_left_click_time = now

        # Scrolling: index + middle up, ring+pinky down, vertical movement
        scrolling_pose = (
            fingers["index"]
            and fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
            and dist_index_thumb > pinch_threshold * 1.1
        )

        if scrolling_pose:
            current_y = (index_tip[1] + middle_tip[1]) / 2
            if self.prev_scroll_y is not None and now - self.last_scroll_time > 0.05:
                delta = self.prev_scroll_y - current_y  # up movement => positive scroll
                scroll_amount = int(delta * (scroll_sensitivity / 25))
                if abs(scroll_amount) >= 1 and self.control_enabled:
                    pyautogui.scroll(scroll_amount)
                    self.last_gesture = "Scrolling"
                    self.last_scroll_time = now
            self.prev_scroll_y = current_y
        else:
            self.prev_scroll_y = None

        # Visual helper circles
        cv2.circle(frame, thumb_tip, 8, (0, 255, 255), -1)
        cv2.circle(frame, index_tip, 8, (0, 255, 0), -1)
        cv2.circle(frame, middle_tip, 8, (255, 255, 0), -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")

        cv2.namedWindow("Gesture Mouse Control")
        cv2.createTrackbar("Sensitivity", "Gesture Mouse Control", 50, 100, lambda _: None)

        print("Controls:")
        print("  c -> toggle cursor control ON/OFF")
        print("  q -> quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            sensitivity = cv2.getTrackbarPos("Sensitivity", "Gesture Mouse Control")

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
                    self._process_hand(frame, hand_lms, sensitivity)
            else:
                self.last_gesture = "No Hand Detected"
                if self.dragging and self.control_enabled:
                    # Safety: release drag if tracking is lost.
                    pyautogui.mouseUp()
                    self.dragging = False

            # FPS + status text
            now = time.time()
            fps = 1.0 / max(1e-6, now - self.prev_frame_time)
            self.prev_frame_time = now

            status_text = f"Control: {'ON' if self.control_enabled else 'OFF'} | Gesture: {self.last_gesture}"
            cv2.putText(frame, status_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                frame,
                "Keys: c=toggle control, q=quit",
                (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Gesture Mouse Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                self.control_enabled = not self.control_enabled
                self.last_gesture = "Control Enabled" if self.control_enabled else "Control Disabled"
                time.sleep(0.12)
            elif key == ord("q"):
                break

        if self.dragging and self.control_enabled:
            pyautogui.mouseUp()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    controller = GestureMouseController()
    controller.run()
