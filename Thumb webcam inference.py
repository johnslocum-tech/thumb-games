"""
Thumb Angle Regressor -- Live Webcam Preview (Two Hands)
============================================================
Shows your live webcam feed with up to two hands tracked at once:
    - a hand on the LEFT side of the frame gets a BLUE box
    - a hand on the RIGHT side of the frame gets a RED box
Each box is labeled with its own live 0.0 (thumb down) - 1.0
(thumb up) value. Handy for testing two-hand control before wiring
it into a game.

Uses thumb_control.py, so it reflects exactly what any game built on
that module will see.

HOW TO USE
-----------
1. Make sure thumb_regressor.pth and thumb_control.py are in the
   same folder as this script.
2. Run this script in Thonny (F5).
3. Show one or both hands to the webcam -- boxes + values appear
   automatically as hands are detected.
4. Press 'q' with the video window focused to quit.
"""

import cv2

from thumb_control import ThumbController

CAMERA_WAIT_MS = 1

BLUE = (235, 150, 70)   # BGR for OpenCV -- displays as blue
RED = (70, 70, 230)     # BGR for OpenCV -- displays as red
WHITE = (255, 255, 255)


def draw_hand_overlay(frame, side, box, value, detected):
    color = BLUE if side == "left" else RED
    label = "LEFT" if side == "left" else "RIGHT"

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        if detected:
            text = f"{label}: {value:.2f}"
        else:
            text = f"{label}: --"

        text_y = max(20, y1 - 10)
        cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_side_bar(frame, side, value, detected, x_offset):
    h, w = frame.shape[:2]
    color = BLUE if side == "left" else RED
    label = "LEFT" if side == "left" else "RIGHT"

    bar_x = x_offset
    bar_y = 60
    bar_w, bar_h = 30, 180

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 2)

    if detected:
        fill_h = int(bar_h * value)
        fill_top = bar_y + (bar_h - fill_h)
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + bar_w, bar_y + bar_h), color, -1)
        value_text = f"{value:.2f}"
    else:
        value_text = "--"

    cv2.putText(frame, label, (bar_x - 5, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, value_text, (bar_x - 5, bar_y + bar_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main():
    print("[main] Loading model and starting webcam thread...")
    controller = ThumbController()
    controller.start()
    print("[main] Ready. Press 'q' in the video window to quit.")

    while True:
        frame = controller.get_latest_frame()
        if frame is None:
            # model/camera still starting up
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            continue

        display_frame = frame.copy()
        boxes = controller.get_hand_boxes()

        left_value = controller.get_left_value()
        right_value = controller.get_right_value()
        left_detected = controller.get_left_detected()
        right_detected = controller.get_right_detected()

        draw_hand_overlay(display_frame, "left", boxes["left"], left_value, left_detected)
        draw_hand_overlay(display_frame, "right", boxes["right"], right_value, right_detected)

        draw_side_bar(display_frame, "left", left_value, left_detected, x_offset=20)
        draw_side_bar(display_frame, "right", right_value, right_detected, x_offset=display_frame.shape[1] - 60)

        cv2.imshow("Thumb Control -- Two Hand Preview", display_frame)

        if cv2.waitKey(CAMERA_WAIT_MS) & 0xFF == ord('q'):
            break

    controller.stop()
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    main()