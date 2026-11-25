"""List usable cameras; settings: vision_tracking/config.json."""
from watercraft.config import load

def main():
    import sys
    import cv2
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY  # as watercraft.deploy.vision.open_camera
    for index in range(load('vision_tracking')['camera_scan_count']):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                print('Camera', index, f'({frame.shape[1]}x{frame.shape[0]} by default)' if ok else '(no frames)')
        finally:
            capture.release()

if __name__ == '__main__':
    main()
