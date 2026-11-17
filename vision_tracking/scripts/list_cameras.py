"""List usable cameras; settings: vision_tracking/config.json."""
from watercraft.config import load

def main():
    import cv2
    for index in range(load('vision_tracking')['camera_scan_count']):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                print('Camera', index)
        finally:
            capture.release()

if __name__ == '__main__':
    main()
