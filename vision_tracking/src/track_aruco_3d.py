"""Measure a rectangle from four known-size ArUco markers.

IDs: 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left.
Place markers on one flat plane. Width/height are distances between marker
centers, not outside edges. At the rectangle center, +X follows ID0 -> ID1,
+Y follows ID0 -> ID3, and +Z is the right-handed perpendicular X cross Y.
Camera coordinates follow OpenCV: X right, Y down, Z away from the camera.
All physical lengths and translation vectors are in meters.
Only marker size and camera intrinsics provide metric scale; no rectangle
width or height is assumed. The camera and rectangle must stay fixed after
calibration; press R to recalibrate if either moves.
"""
from pathlib import Path
import time
import cv2
import numpy as np

CAM_INDEX = 0
CORNER_IDS = (0, 1, 2, 3)
TARGET_ID = 4
MARKER_LENGTH_M = 0.04
N_CALIB_FRAMES = 20
AXIS_LENGTH_M = 0.05
POSE_PRINT_INTERVAL_S = 1.0
CALIB_FILE = Path(__file__).with_name('camera_calibration.npz')


def marker_object_points(length_m):
    half = length_m / 2.0
    # Required order for SOLVEPNP_IPPE_SQUARE.
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def detect_and_draw_markers(frame_bgr, detector):
    """Return an annotated copy and ID -> four image corners (pixels)."""
    vis = frame_bgr.copy()
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    found = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
    return vis, found


def estimate_marker_pose(corners, K, D, length_m):
    """Return T_camera_from_marker (4x4), with translation in meters.

    Marker axes: +X toward its right edge, +Y toward its top edge,
    +Z out of the printed face, following the decoded marker orientation.

    Evaluate both planar pose candidates and choose the positive-depth pose
    with the smallest corner reprojection error.
    """
    obj = marker_object_points(length_m)
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        obj, pixels, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    candidates = []
    for rv, tv in zip(rotations, translations):
        R, _ = cv2.Rodrigues(rv)
        camera_corners = (R @ obj.T + tv.reshape(3, 1)).T
        if not np.isfinite(camera_corners).all() or np.any(camera_corners[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(obj, rv, tv, K, D)
        error = np.mean(np.sum((projected.reshape(4, 2) - pixels) ** 2, axis=1))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R
        transform[:3, 3] = tv.reshape(3)
        if np.isfinite(error):
            candidates.append((error, transform))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def estimate_marker_center(corners, K, D, length_m):
    """Return the marker center in camera coordinates (meters), or None."""
    pose = estimate_marker_pose(corners, K, D, length_m)
    return None if pose is None else pose[:3, 3].copy()


def locate_aruco_4(id_to_corners, geometry, K, D, length_m=MARKER_LENGTH_M):
    """Locate ID4 and return its complete pose relative to camera and origin.

    Use D=0 for corners detected in an undistorted image. Positions are meters.
    T_a_from_b maps homogeneous coordinates from frame b into frame a.
    Rotation columns are marker +X/+Y/+Z directions in the origin frame;
    +X is the heading convention, not a measured direction of travel.
    heading_angle_rad/deg measures that heading from origin +X toward +Y
    in the rectangle plane, in [-pi, pi] / [-180, 180]. It is None when the
    heading is perpendicular to the plane and has no defined planar angle.
    inside_rectangle tests the center's projection onto the measured rectangle
    plane, so the marker may be above the plane. Poses outside are returned
    with inside_rectangle=False rather than silently discarded.
    """
    if TARGET_ID not in id_to_corners or geometry is None:
        return None
    T_camera_from_id4 = estimate_marker_pose(
        id_to_corners[TARGET_ID], K, D, length_m)
    if T_camera_from_id4 is None:
        return None
    T_origin_from_id4 = geometry['T_origin_from_camera'] @ T_camera_from_id4
    rotation = T_origin_from_id4[:3, :3]
    # Heading is the marker's local +X projected into the rectangle XY plane.
    # Zero is rectangle +X; positive angles turn toward rectangle +Y.
    heading = rotation[:, 0]
    heading_angle_rad = (float(np.arctan2(heading[1], heading[0]))
                         if np.linalg.norm(heading[:2]) > 1e-8 else None)
    # Fit rectangle-plane coordinates without assuming camera-aligned edges.
    bl, br, tr, tl = geometry['corners']
    right = ((br - bl) + (tr - tl)) / 2.0
    up = ((tl - bl) + (tr - br)) / 2.0
    uv, _, _, _ = np.linalg.lstsq(
        np.column_stack((right, up)),
        T_camera_from_id4[:3, 3] - geometry['center'], rcond=None)
    return {
        'T_camera_from_id4': T_camera_from_id4,
        'T_origin_from_id4': T_origin_from_id4,
        'position_origin_m': T_origin_from_id4[:3, 3].copy(),
        'rotation_camera_from_id4': T_camera_from_id4[:3, :3].copy(),
        'rotation_origin_from_id4': rotation.copy(),
        'heading_origin': heading.copy(),
        'heading_angle_rad': heading_angle_rad,
        'heading_angle_deg': (None if heading_angle_rad is None
                              else float(np.degrees(heading_angle_rad))),
        'up_origin': rotation[:, 1].copy(),
        'normal_origin': rotation[:, 2].copy(),
        'inside_rectangle': bool(np.all(np.abs(uv) <= 0.5 + 1e-6)),
    }


def rectangle_from_centers(centers):
    """Measure BL, BR, TR, TL and construct the centered rectangle frame.

    Opposite edge lengths are averaged; dimensions are never predetermined.
    R maps rectangle coordinates into OpenCV camera coordinates. X follows
    ID0 -> ID1 exactly. Y follows ID0 -> ID3 after removing measurement error
    along X, making the rotation orthonormal. Z = X cross Y.
    """
    points = np.asarray(centers, dtype=np.float64).reshape(4, 3)
    if not np.isfinite(points).all():
        raise ValueError('Marker centers must be finite.')
    bl, br, tr, tl = points
    right = br - bl
    forward = tl - bl
    width = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2.0
    height = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2.0
    if min(np.linalg.norm(right), np.linalg.norm(forward)) < 1e-6:
        raise ValueError('Corner marker centers overlap.')
    normal = np.cross(right, forward)
    if np.linalg.norm(normal) < 1e-6 * np.linalg.norm(right) * np.linalg.norm(forward):
        raise ValueError('Corner marker centers are collinear.')
    x_axis = right / np.linalg.norm(right)
    z_axis = normal / np.linalg.norm(normal)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    center = points.mean(axis=0)
    T_camera_from_origin = np.eye(4, dtype=np.float64)
    T_camera_from_origin[:3, :3] = R
    T_camera_from_origin[:3, 3] = center
    T_origin_from_camera = np.eye(4, dtype=np.float64)
    T_origin_from_camera[:3, :3] = R.T
    T_origin_from_camera[:3, 3] = -R.T @ center
    return {'center': center, 'R': R,
            'T_camera_from_origin': T_camera_from_origin,
            'T_origin_from_camera': T_origin_from_camera,
            'width': float(width), 'height': float(height), 'corners': points}


def find_origin(id_to_corners, K, D, length_m=MARKER_LENGTH_M):
    """Estimate rectangle geometry for one frame, or None if a marker is missing.

    The returned dictionary includes T_camera_from_origin (4x4), satisfying
    [p_camera, 1] = T_camera_from_origin @ [p_origin, 1], in meters.
    T_origin_from_camera provides the inverse mapping. The origin axes are
    fitted to the rectangle: X follows ID0 -> ID1, Y toward ID3, Z = X cross Y.
    Call with zero distortion when corners come from an undistorted frame.
    """
    if not all(i in id_to_corners for i in CORNER_IDS):
        return None
    centers = [estimate_marker_center(id_to_corners[i], K, D, length_m) for i in CORNER_IDS]
    if any(center is None for center in centers):
        return None
    return rectangle_from_centers(centers)


def draw_origin_axes(vis, geometry, K, D):
    rvec, _ = cv2.Rodrigues(geometry['R'])
    center = geometry['center'].reshape(3, 1)
    if center[2, 0] <= AXIS_LENGTH_M:
        return  # Keep all drawn endpoints in front of the camera.
    origin_pixel, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, center, K, D)
    origin_pixel = tuple(np.rint(origin_pixel.reshape(2)).astype(int))
    cv2.circle(vis, origin_pixel, 5, (255, 255, 255), -1)
    cv2.putText(vis, 'Origin', (origin_pixel[0] + 8, origin_pixel[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.drawFrameAxes(vis, K, D, rvec, center, AXIS_LENGTH_M, 3)
    endpoints, _ = cv2.projectPoints(np.eye(3) * AXIS_LENGTH_M, rvec, center, K, D)
    # Depth can project to a dot at the optical axis; keep its meaning visible.
    cv2.putText(vis, 'X: ID0 -> ID1 | Y: ID0 -> ID3 | Z: X cross Y', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for label, pixel, color in zip(('X', 'Y', 'Z'), endpoints.reshape(3, 2),
                                   ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.putText(vis, label, tuple(np.rint(pixel).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def display_id4_rotation(vis, pose):
    """Display the full marker-to-origin rotation matrix (dimensionless)."""
    cv2.putText(vis, 'R_origin_from_id4 (marker axes in origin frame):',
                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for index, row in enumerate(pose['rotation_origin_from_id4']):
        text = '[ ' + '  '.join(f'{value: .4f}' for value in row) + ' ]'
        cv2.putText(vis, text, (10, 265 + 25 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_id4_pose(pose):
    """Print orientation matrices and full transforms; translations are meters."""
    angle = pose['heading_angle_deg']
    print('ID4 angle from origin +X: ' +
          ('undefined (heading normal to plane)' if angle is None else f'{angle:+.2f} deg'))
    for name in ('rotation_camera_from_id4', 'rotation_origin_from_id4',
                 'T_camera_from_id4', 'T_origin_from_id4'):
        print(f'{name}:')
        print(np.array2string(pose[name], precision=5, suppress_small=True))


def main():
    if MARKER_LENGTH_M <= 0:
        raise ValueError('MARKER_LENGTH_M must be positive')
    with np.load(CALIB_FILE) as calibration:
        K, D = calibration['K'], calibration['D']
    zero_distortion = np.zeros_like(D)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    capture = cv2.VideoCapture(CAM_INDEX)
    samples, geometry, preview = [], None, None
    last_pose_print = float('-inf')
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera.')
        print('Keep all four corner markers visible and still. R recalibrates; Esc quits.')
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Cannot read camera frame.')
            # Keep the same K after undistortion; all subsequent poses use D=0.
            undistorted = cv2.undistort(frame, K, D)
            vis, found = detect_and_draw_markers(undistorted, detector)
            if geometry is None:
                measured = find_origin(found, K, zero_distortion, MARKER_LENGTH_M)
                if measured is not None:
                    samples.append(measured['corners'])
                    preview = rectangle_from_centers(np.median(samples, axis=0))
                if len(samples) >= N_CALIB_FRAMES:
                    geometry = rectangle_from_centers(np.median(samples, axis=0))
                    print(f"Width: {geometry['width']:.4f} m; height: {geometry['height']:.4f} m")
                    print(f"Origin in camera coordinates (m): {geometry['center']}")
                    print('T_camera_from_origin (translation in m):')
                    print(geometry['T_camera_from_origin'])
                    print('T_origin_from_camera (inverse; translation in m):')
                    print(geometry['T_origin_from_camera'])
                missing = [i for i in CORNER_IDS if i not in found]
                status = f'Calibration: {len(samples)}/{N_CALIB_FRAMES}'
                if missing:
                    status += f' | missing IDs: {missing}'
                elif measured is None:
                    status += ' | marker pose failed'
                else:
                    status += ' | preliminary estimate'
                cv2.putText(vis, status, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(vis, f'Detected IDs: {sorted(found)}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            # Show estimates immediately while collecting the calibration samples.
            displayed_geometry = geometry if geometry is not None else preview
            if displayed_geometry is not None:
                draw_origin_axes(vis, displayed_geometry, K, zero_distortion)
                cv2.putText(vis, f"Width {displayed_geometry['width']:.4f} m  Height {displayed_geometry['height']:.4f} m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                id4_pose = locate_aruco_4(found, displayed_geometry, K, zero_distortion)
                if id4_pose is not None:
                    display_id4_rotation(vis, id4_pose)
                    now = time.monotonic()
                    if now - last_pose_print >= POSE_PRINT_INTERVAL_S:
                        print_id4_pose(id4_pose)
                        last_pose_print = now
                    T_camera_from_id4 = id4_pose['T_camera_from_id4']
                    id4_rvec, _ = cv2.Rodrigues(T_camera_from_id4[:3, :3])
                    cv2.drawFrameAxes(vis, K, zero_distortion, id4_rvec,
                                      T_camera_from_id4[:3, 3], MARKER_LENGTH_M / 2, 2)
                    xyz = id4_pose['position_origin_m']
                    cv2.putText(vis, f'ID4: X {xyz[0]:.4f} Y {xyz[1]:.4f} Z {xyz[2]:.4f} m',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    angle = id4_pose['heading_angle_deg']
                    angle_text = ('ID4 angle from +X: undefined (heading normal to plane)'
                                  if angle is None else f'ID4 angle from +X: {angle:+.2f} deg')
                    cv2.putText(vis, angle_text, (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(vis, f"ID4 inside rectangle: {id4_pose['inside_rectangle']}",
                                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, 'ID4 not detected or pose unavailable', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow('ArUco measured rectangle', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('r'), ord('R')):
                samples, geometry, preview = [], None, None
                last_pose_print = float('-inf')
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
