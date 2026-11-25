"""Preview and capture frames; settings: configs/deploy.json and vision_tracking/config.json."""
from datetime import datetime
from watercraft.config import load, path


def main():
    import cv2
    from watercraft.deploy.vision import open_camera
    real, tools = load('deploy'), load('vision_tracking')
    capture = open_camera(real['camera_index'])
    try:
        cv2.namedWindow(tools['preview_window'], cv2.WINDOW_NORMAL)
        print('S: save frame; Q/Esc: quit. Motors are not opened.')
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Cannot read camera')
            cv2.imshow(tools['preview_window'], frame)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                directory = path(tools['capture_directory'])
                directory.mkdir(parents=True, exist_ok=True)
                filename = directory / (datetime.now().strftime('calibration_%Y%m%d_%H%M%S_%f') + '.jpg')
                if not cv2.imwrite(str(filename), frame):
                    raise OSError(f'Could not save {filename}')
                print(filename)
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
