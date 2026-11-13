"""ChArUco calibration from configured images, with optional reprojection verification."""
from watercraft.config import load, path

def main():
    import cv2
    import glob
    import numpy as np
    from datetime import datetime
    c, real = load('tools'), load('real')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(tuple(c['board_squares']), c['square_length_m'], c['board_marker_length_m'], dictionary)
    board.setLegacyPattern(c['legacy_pattern'])
    corners, ids, size = [], [], None
    for filename in sorted(glob.glob(str(path(c['calibration_images'])))):
        frame = cv2.imread(filename)
        if frame is None:
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if size is not None and current_size != size:
            raise ValueError('Calibration images have different resolutions')
        markers, marker_ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dictionary)
        if marker_ids is None:
            continue
        count, points, point_ids = cv2.aruco.interpolateCornersCharuco(markers, marker_ids, frame, board)
        if count is not None and count >= 4:
            corners.append(points.astype(np.float32))
            ids.append(point_ids.astype(np.int32))
            size = current_size
    if not corners:
        raise ValueError('No usable ChArUco corners in configured calibration images')
    error, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(corners, ids, board, size, None, None)
    output = path(real['calibration_file'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        import shutil
        shutil.copy2(output, output.with_name(output.stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + output.suffix))
    np.savez(output, K=K, D=D, image_size=size, projection_error=error)
    print('Saved', output, 'reprojection error:', error)
    test = cv2.imread(str(path(c['calibration_test_image'])))
    if test is not None:
        markers, marker_ids, _ = cv2.aruco.detectMarkers(test, dictionary)
        if marker_ids is not None:
            count, points, point_ids = cv2.aruco.interpolateCornersCharuco(markers, marker_ids, test, board)
            if count is not None and count >= 4:
                obj = board.getChessboardCorners()[point_ids.flatten()]
                ok, rotation, translation = cv2.solvePnP(obj, points, K, D)
                if ok:
                    projected, _ = cv2.projectPoints(obj, rotation, translation, K, D)
                    for i, corner_id in enumerate(point_ids.flatten()):
                        if int(corner_id) in c['verify_corner_ids']:
                            print('Corner', corner_id, 'error (px):', float(np.linalg.norm(projected[i] - points[i])))

if __name__ == '__main__':
    main()
