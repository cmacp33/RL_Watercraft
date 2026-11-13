"""List usable cameras; settings: configs/tools.json."""
from watercraft.config import load

def main():
    import cv2
    for index in range(load('tools')['camera_scan_count']):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                print('Camera', index)
        finally:
            capture.release()

if __name__ == '__main__':
    main()
