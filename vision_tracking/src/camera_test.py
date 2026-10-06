import cv2
import os
from datetime import datetime   

# USB camera 0 because theres only 1
cap = cv2.VideoCapture(0)

output_folder = "/Users/tracysun/Projects/fiducial_tracking/markers/Calibration_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# keep reading frame
last_key = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("USB Camera", frame) # stream frames

    # Read key once per loop iteration to avoid consuming the event twice
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    # On rising edge of '1' key, save an image (allows multiple saves)
    if key == ord('1') and last_key != ord('1'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"calibration_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved image to {filename}")

    # update last_key (reset when no key is pressed)
    if key == 255:
        last_key = None
    else:
        last_key = key
        


cap.release()
cv2.destroyAllWindows()

