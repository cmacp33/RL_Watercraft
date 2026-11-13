"""Show a preview from each camera index so you can tell which one is the USB camera.

Run from vision_tracking/:  python src/list_cameras.py
Press any key to move to the next camera.
"""
import sys

import cv2

backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
for index in range(5):
    cap = cv2.VideoCapture(index, backend)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        print(f"camera {index}: not available")
        cap.release()
        continue
    h, w = frame.shape[:2]
    print(f"camera {index}: {w}x{h}  (window open, press any key for the next one)")
    cv2.putText(frame, f"--cam {index}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("camera preview", frame)
    cv2.waitKey(0)
    cap.release()
cv2.destroyAllWindows()
