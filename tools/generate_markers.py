"""Generate the configured marker IDs without overwriting existing files."""
from watercraft.config import load, path

def main():
    import cv2
    config = load('tools')
    output = path(config['marker_directory'])
    output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id in config['marker_ids']:
        filename = output / f'aruco_{marker_id}.png'
        if filename.exists():
            print('Preserved existing marker:', filename)
            continue
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, config['marker_pixels'])
        if not cv2.imwrite(str(filename), image):
            raise OSError(f'Could not write {filename}')
        print(filename)

if __name__ == '__main__':
    main()
