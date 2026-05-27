"""
Vision processor for Project Vertex v4.
Runs MediaPipe hand landmarking, feeds GestureEngine, annotates frame for PiP.
Returns fingertip pixel positions for ambient trail rendering.
"""
import cv2
import mediapipe as mp
import math
import time
import os

from gesture_engine import GestureEngine, GestureMode, MODE_META


class HandSensor:
    def __init__(self, detection_con=0.7, track_con=0.7, min_palm_px=35):
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core import base_options as ba

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        opts = HandLandmarkerOptions(
            base_options=ba.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )
        self.landmarker = HandLandmarker.create_from_options(opts)
        self.engine     = GestureEngine()

        # BGR drawing colours
        self._cyan   = (255, 200,   0)
        self._white  = (255, 255, 255)
        self._orange = (  0, 160, 255)
        self._red    = (  0,   0, 220)

        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]

        self.min_palm_px    = min_palm_px
        self.prev_centers:  dict[str, tuple] = {}
        self._pinch_state:  dict[str, bool]  = {}   # per-hand hysteresis

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_frame(self, img):
        """Process one webcam frame. Returns (annotated_bgr, data_dict)."""
        img  = cv2.flip(img, 1)
        h, w = img.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp_img)

        now  = time.time()
        data = {
            "hands_detected": 0,
            "both_hands":     [],
            "zoom_distance":  None,
            "fist_reset":     False,
            "fist_progress":  0.0,
            "cam_size":       (w, h),
        }

        if not result.hand_landmarks:
            # Clear per-hand state so stale values don't persist across gaps
            self._pinch_state.clear()
            self.prev_centers.clear()
            return img, data

        data["hands_detected"] = len(result.hand_landmarks)
        all_hands = []

        for idx, (landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)):

            label   = handedness[0].category_name   # "Left" / "Right"
            hand_id = f"{label}_{idx}"

            lm_list = [[i, int(lm.x*w), int(lm.y*h), lm.x, lm.y, lm.z]
                       for i, lm in enumerate(landmarks)]

            mode = self.engine.get_mode(hand_id, lm_list)

            if self.engine.check_fist_reset(hand_id, lm_list, now):
                data["fist_reset"] = True
            data["fist_progress"] = max(
                data["fist_progress"],
                self.engine.fist_hold_progress(hand_id, now))

            cx = sum(p[1] for p in lm_list) // 21
            cy = sum(p[2] for p in lm_list) // 21
            prev = self.prev_centers.get(hand_id, (cx, cy))
            velocity = (cx - prev[0], cy - prev[1])
            self.prev_centers[hand_id] = (cx, cy)

            ix, iy = lm_list[8][1], lm_list[8][2]
            tx, ty = lm_list[4][1], lm_list[4][2]
            pinch_dist = math.hypot(ix-tx, iy-ty)
            # Hysteresis: latch ON at <60px, latch OFF only once gap > 80px.
            # 60px works at typical arm-length camera distance (640x480).
            was_pinched = self._pinch_state.get(hand_id, False)
            pinched = pinch_dist < (80 if was_pinched else 60)
            self._pinch_state[hand_id] = pinched

            # All 5 fingertip pixel positions for ambient trail rendering
            fingertips = [
                (lm_list[4][1],  lm_list[4][2]),   # thumb
                (lm_list[8][1],  lm_list[8][2]),   # index
                (lm_list[12][1], lm_list[12][2]),  # middle
                (lm_list[16][1], lm_list[16][2]),  # ring
                (lm_list[20][1], lm_list[20][2]),  # pinky
            ]

            finger_count = self.engine.count_extended_fingers(lm_list)

            # Palm size = wrist (0) → middle-finger MCP (9) distance in pixels.
            # Shrinks as hand moves away — reliable proxy for depth.
            wx, wy = lm_list[0][1], lm_list[0][2]
            mx, my = lm_list[9][1], lm_list[9][2]
            palm_px = math.hypot(mx - wx, my - wy)
            too_far = palm_px < self.min_palm_px

            hand_info = {
                "id":           hand_id,
                "label":        label,
                "mode":         mode,
                "center":       (cx, cy),
                "velocity":     velocity,
                "lm_list":      lm_list,
                "index_tip":    (ix, iy),
                "thumb_tip":    (tx, ty),
                "pinched":      pinched,
                "fingertips":   fingertips,
                "finger_count": finger_count,
                "fist_progress":self.engine.fist_hold_progress(hand_id, now),
                "palm_px":      round(palm_px, 1),
                "too_far":      too_far,
            }
            all_hands.append(hand_info)

            self._draw_skeleton(img, landmarks, w, h)
            self._draw_mode_label(img, mode, cx, cy, pinched,
                                  self.engine.fist_hold_progress(hand_id, now),
                                  too_far=too_far)

        data["both_hands"] = all_hands

        if len(all_hands) == 2:
            p1 = all_hands[0]["index_tip"]
            p2 = all_hands[1]["index_tip"]
            data["zoom_distance"] = math.hypot(p1[0]-p2[0], p1[1]-p2[1])

        return img, data

    # ------------------------------------------------------------------
    # Annotation drawing (onto cv2 frame for PiP)
    # ------------------------------------------------------------------

    def _draw_skeleton(self, img, landmarks, w, h):
        for a, b in self.HAND_CONNECTIONS:
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x*w), int(landmarks[a].y*h))
                p2 = (int(landmarks[b].x*w), int(landmarks[b].y*h))
                cv2.line(img, p1, p2, self._cyan, 1)
        for lm in landmarks:
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 3, self._white, cv2.FILLED)
            cv2.circle(img, (cx, cy), 3, self._cyan,  1)

    def _draw_mode_label(self, img, mode, cx, cy, pinched, fist_prog, too_far=False):
        if too_far:
            cv2.circle(img, (cx, cy), 18, self._red, 2)
            cv2.putText(img, "FAR", (cx - 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self._red, 2)
            return

        label, rgb = MODE_META[mode]
        bgr = (rgb[2], rgb[1], rgb[0])

        if fist_prog > 0.05:
            angle = int(360 * fist_prog)
            cv2.ellipse(img, (cx, cy), (26, 26), -90, 0, angle, self._orange, 3)
            cv2.putText(img, f"{int(fist_prog*100)}%",
                        (cx-20, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._orange, 2)
        else:
            cv2.circle(img, (cx, cy), 16 if mode == GestureMode.NONE else 20, bgr, 2)

        cv2.putText(img, label, (cx-40, cy-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, bgr, 2)

        if pinched:
            cv2.circle(img, (cx, cy), 9, self._red, cv2.FILLED)
