import cv2
import numpy as np
from collections import defaultdict
import math
import serial
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_contract import NumpyPolicy, POSITION_SCALE_M, PoseObservationAdapter, esp32_motor_action

# ====== SERIAL Settings
from watercraft.config import load as load_config, path as config_path
_SETTINGS = load_config("real")

SERIAL_PORT = _SETTINGS['serial_port']
BAUD_RATE = _SETTINGS['baud_rate']
CONTROL_PERIOD_S = _SETTINGS['control_period_s']

# ===== INPUT =====
CAM_INDEX = _SETTINGS['camera_index']
DICT = cv2.aruco.DICT_4X4_50
CALIB_FILE = str(config_path(_SETTINGS["calibration_file"]))

TARGET = tuple(v * 100 for v in _SETTINGS["goal"])
STOP_DISTANCE_CM = _SETTINGS["goal_radius_m"] * 100
HEADING_TOL_DEG = _SETTINGS['heading_tolerance_deg']
FORWARD_CMD = _SETTINGS['forward_command']
TURN_CMD_MIN = _SETTINGS['turn_command_min']
TURN_CMD_MAX = _SETTINGS['turn_command_max']
TURN_KP = _SETTINGS['turn_kp']

# take the average of how many average frames to use for calibration
N_CALIB_FRAMES = _SETTINGS["legacy_calibration_frames"]

FRAME_W, FRAME_H = _SETTINGS["frame_width"], _SETTINGS["frame_height"]

ID_BL = _SETTINGS["corner_ids"][0]
ID_BR = _SETTINGS["corner_ids"][1]
ID_TR = _SETTINGS["corner_ids"][2]
ID_TL = _SETTINGS["corner_ids"][3]

ID_INSIDE = _SETTINGS['target_id']
WATER_ID = _SETTINGS['target_id']

W = _SETTINGS["rim_width_cm"]
H = _SETTINGS["rim_height_cm"]

CORNER_IDS = list(_SETTINGS["corner_ids"])

MARKER_LENGTH_CM = _SETTINGS["marker_length_m"] * 100

# Define positioning of A singular marker
_half = MARKER_LENGTH_CM / 2.0
MARKER_OBJ_PTS = np.array([
    [-_half,  _half, 0],   # top-left
    [ _half,  _half, 0],   # top-right
    [ _half, -_half, 0],   # bottom-right
    [-_half, -_half, 0],   # bottom-left
], dtype=np.float64)

# Real world Tank Rim corner locations
RIM_WORLD_PTS_3D = np.array([
    [0.0, 0.0, 0.0],  # BL
    [W,   0.0, 0.0],  # BR
    [W,   H,   0.0],  # TR
    [0.0, H,   0.0],  # TL
], dtype=np.float32)


def marker_center(corners_4x2: np.ndarray) -> np.ndarray:
    '''
    Finds the center of aruco
    '''
    return corners_4x2.mean(axis=0) # for x,y coord

def detect_and_draw_markers(frame_bgr, detector):
    '''
    Detect aruco markers and draw them
    Returns a dictionary of marker id to its 4 corners
    '''
    
    frame_out = frame_bgr.copy()
    # grayscale to help with detection
    gray = cv2.cvtColor(frame_out, cv2.COLOR_BGR2GRAY)
    
    # returns 2 lists of corners and ids
    corners, ids, _ = detector.detectMarkers(gray)

    # initiate a list to pair up ids to corners
    id_to_corners = {}

    # match ids to the pixel location
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_out, corners, ids)

        # Builds a look up table: marker_id -> corners (4,2)
        for c, marker_id in zip(corners, ids.flatten()): # flatten 
            id_to_corners[int(marker_id)] = c[0]  # c is (1,4,2), so now if we have c[0] we only have (4,2) left
            # 4 indicates the 4 corners of the marker, and 2 indicates the (x,y) in terms of pixel position

        # Draw center dots + labels
        for marker_id, pts in id_to_corners.items():
            center = marker_center(pts).astype(int)
            cv2.circle(frame_out, tuple(center), 5, (0, 255, 0), -1)
            cv2.putText(frame_out, f"ID {marker_id}", (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame_out, id_to_corners


def draw_rectangle_edges(frame, img_pts_bl_br_tr_tl):
    """Draw rectangle edges on the frame using [BL, BR, TR, TL] order."""
    pts = img_pts_bl_br_tr_tl.astype(int)
    # BL->BR->TR->TL->BL
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 0, 255), 2)

def find_homography_mat(cap, detector, K, D):
    '''
    Collects N_CALIB_FRAMES frames where all 4 corner markers are visible,
    averages their pixel centers, then computes and returns H and img_pts.
    '''

    # each key is a corner ID, value is a growing list of (x,y) center observations
    accumulated = defaultdict(list)

    print(f"Collecting {N_CALIB_FRAMES} frames for homography. Make sure all 4 corners are visible.")

    # keep going until every corner has N_CALIB_FRAMES observations
    while any(len(accumulated[cid]) < N_CALIB_FRAMES for cid in CORNER_IDS):

        ret, frame = cap.read()
        if not ret:
            continue

        # undistort before detection — fixes lens distortion
        corr_frame = cv2.undistort(frame, K, D)
        _, id_to_corners = detect_and_draw_markers(corr_frame, detector)

        # only use this frame if all 4 corner markers are detected
        if not all(cid in id_to_corners for cid in CORNER_IDS):
            missing = [cid for cid in CORNER_IDS if cid not in id_to_corners]
            print(f"  Skipping frame — missing IDs: {missing}")
            continue

        # accumulate the center of each corner marker
        for cid in CORNER_IDS:
            accumulated[cid].append(marker_center(id_to_corners[cid]))

        counts = {cid: len(accumulated[cid]) for cid in CORNER_IDS}
        print(f"  Collected: {counts}")

    # average the observations and report stability
    print("\nDetection stability (std should be < 1.0 px):")
    
    # no point in printing accumulated
    # print(accumulated)
    avg_centers = {}
    for cid in CORNER_IDS:
        pts = np.array(accumulated[cid])  # (N, 2)
        avg = pts.mean(axis=0)
        std = pts.std(axis=0)
        avg_centers[cid] = avg
        print(f"  ID {cid}: mean=({avg[0]:.1f}, {avg[1]:.1f})  std=({std[0]:.3f}, {std[1]:.3f})")

    # image points in [BL, BR, TR, TL] order — averaged pixel centers
    img_pts = np.array([
        avg_centers[ID_BL],
        avg_centers[ID_BR],
        avg_centers[ID_TR],
        avg_centers[ID_TL],
    ], dtype=np.float32)

    # corresponding real-world positions in cm — must match the order above
    rect_pts = np.array([
        [0.0, 0.0],  # BL
        [W,   0.0],  # BR
        [W,   H  ],  # TR
        [0.0, H  ],  # TL
    ], dtype=np.float32)

    Hmat, _ = cv2.findHomography(img_pts, rect_pts)
    print(f"\nHomography computed:\n{Hmat}\n")
    return Hmat, img_pts



def map_to_rect(id_to_corners, marker_id, Hmat):
    '''
    Takes the center of marker_id in pixel space and maps it to
    real-world cm coordinates using H.
    Returns (x_cm, y_cm) or None if marker not detected.
    YAW is measured CCW from +X.
    '''
    if Hmat is None or marker_id not in id_to_corners:
        return None

    p_px = marker_center(id_to_corners[marker_id]).astype(np.float32)  # (x, y) in pixels
    p_rect = cv2.perspectiveTransform(p_px.reshape(1, 1, 2), Hmat)[0, 0]  # (x, y) in cm
    
    
    
    return p_rect

def find_extrinsic_matrix(img_pts, K, D):
    '''
    Find extrinsic matrix [R|t] from averaged corner pixel positions.
    img_pts: numpy array (4,2) of corner pixel positions [BL,BR,TR,TL]
    '''
    
    # img_pts is already (4,2) array of averaged corner positions
    img_pts_2d = img_pts
    world_pts_3d = RIM_WORLD_PTS_3D  # Corresponding world points in same order
    
    # convert to numpy arrays for solvePnP (though they already are)
    img_pts_2d = np.array(img_pts_2d, dtype=np.float32)
    world_pts_3d = np.array(world_pts_3d, dtype=np.float32)
    
    retval, rvec, tvec = cv2.solvePnP(
    objectPoints = world_pts_3d,
    imagePoints =  img_pts_2d,
    cameraMatrix=K, 
    distCoeffs=D, 
    rvec = None, 
    tvec = None,
    useExtrinsicGuess=False,
    flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not retval:
        print("[find_extrinsic_matrix] solvePnP failed")
        return None, None, None

    # convert rvec to R 3x3 
    R, _ = cv2.Rodrigues(rvec)
    tvec    = tvec.flatten()   # (3,1) → (3,)
    
    print(f"[Extrinsic] R:\n{R}")
    print(f"[Extrinsic] t: {tvec}")
    print(f"[Extrinsic] Full [R|t]:\n{np.hstack([R, tvec.reshape(3,1)])}")

    return R, tvec, rvec

def find_water_extrinsic (id_to_corners, K, D, R, t):
    # print(f"5 print D matrix {D}, {type(D)},{D.shape}")
    if WATER_ID not in id_to_corners:
        print(f"Marker ID {WATER_ID} not detected, cannot find water line.")
        return None, None, None
    
    img_pts = id_to_corners[WATER_ID].astype(np.float64)   # (4,2)
    
    ok, water_rvec, water_tvec = cv2.solvePnP(
        MARKER_OBJ_PTS,
        img_pts,
        K, 
        D,
        rvec = None, 
        tvec = None,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not ok:
        print("[find_water_extrinsic] solvePnP on water marker failed")
        return None, None, None

    p_cam = water_tvec.flatten()   # (3,)  [x_cam, y_cam, z_cam]

    # using the rim extrinsic [R|t] in reverse:
    # p_world = Rᵀ · (p_cam - t)
    p_world = R.T @ (p_cam - t)

    # Extract all coordinates in world frame
    x_water = float(p_world[0])
    y_water = float(p_world[1])
    Z_water = float(p_world[2])
    
    r3 = R[:, 2]
    t_water = t + Z_water * r3
    extrinsic_water = np.hstack([R, t_water.reshape(3, 1)])

    print(f"[Water] p_cam:   {p_cam}")
    print(f"[Water] p_world: {p_world}")
    print(f"[Water] Position: x={x_water:.4f}, y={y_water:.4f}, Z={Z_water:.4f} cm")

    return extrinsic_water, t_water, Z_water

def normalize_angle_deg(angle_deg):
    """Normalize angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def marker_yaw_in_water_frame(marker_rvec, R):
    """
    Compute the marker yaw in the water frame.

    Convention:
    - marker local +X axis is used as heading
    - yaw is measured CCW from +X_water in the water XY plane
    """
    R_marker_cam, _ = cv2.Rodrigues(marker_rvec)

    # marker frame -> water frame
    R_water_marker = R.T @ R_marker_cam

    # Marker local +X axis expressed in water frame
    x_axis_water = R_water_marker[:, 0]
    yaw_rad = math.atan2(x_axis_water[1], x_axis_water[0])
    yaw_deg = normalize_angle_deg(math.degrees(yaw_rad))

    return yaw_deg, R_water_marker


def locate_marker_in_water_frame(id_to_corners, marker_id, K, D, R, t_water):
    """
    Return the marker origin in water-frame coordinates (x, y, z)
    and its yaw angle in degrees.
    """
    if marker_id not in id_to_corners:
        return None

    img_pts = id_to_corners[marker_id].astype(np.float64)

    ok, marker_rvec, marker_tvec = cv2.solvePnP(
        MARKER_OBJ_PTS,
        img_pts,
        K,
        D,
        rvec=None,
        tvec=None,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        print(f"[locate_marker_in_water_frame] solvePnP failed for marker {marker_id}")
        return None

    p_cam = marker_tvec.flatten()

    # X_cam = R @ X_water + t_water  ->  X_water = R.T @ (X_cam - t_water)
    p_water = R.T @ (p_cam - t_water)
    xw, yw, zw = map(float, p_water)

    yaw_deg, _ = marker_yaw_in_water_frame(marker_rvec, R)

    return xw, yw, zw, yaw_deg

def water_homography(R, t_water, K):
    
    '''Calculates homography for water plane'''
    
    r1 = R[:, 0]   # world X axis in camera frame
    r2 = R[:, 1]   # world Y axis in camera frame
    # r3 is dropped — we already absorbed Z_water into t_water

    H_water = K @ np.column_stack([r1, r2, t_water])   # (3x3)
    H_water = H_water / H_water[2, 2]                  # normalise so H[2,2]=1

    print(f"[H_water]:\n{H_water}")
    return H_water
    
    
def draw_water_plane_rectangle(vis, H_water):
    """
    Uses H_water inverse to project the 4 known world corners
    back into image pixel coordinates and draws the rectangle.
    """

    # 4 rim corners in world frame (X,Y only — H_water handles Z)
    corners_world = np.array([
        [0.0, 0.0],   # BL
        [W,   0.0],   # BR
        [W,   H  ],   # TR
        [0.0, H  ],   # TL
    ], dtype=np.float32)

    H_inv = np.linalg.inv(H_water)

    # project world corners → image pixels
    img_pts = cv2.perspectiveTransform(
        corners_world.reshape(1, -1, 2), H_inv).reshape(4, 2).astype(int)

    # draw BL→BR→TR→TL→BL
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(vis, tuple(img_pts[a]), tuple(img_pts[b]), (0, 200, 255), 2)

def compute_motor_command_to_target(x_cm, y_cm, yaw_deg, target_xy):
    """
    Turn the boat until its heading points toward the target, then drive straight.

    Returns
    -------
    motor_left, motor_right, debug_info
    """
    dx = float(target_xy[0] - x_cm)
    dy = float(target_xy[1] - y_cm)
    distance_cm = math.hypot(dx, dy)
    target_heading_deg = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
    heading_error_deg = normalize_angle_deg(target_heading_deg - yaw_deg)

    if distance_cm <= STOP_DISTANCE_CM:
        motor_left = 0
        motor_right = 0
        mode = "stop"
    elif abs(heading_error_deg) > HEADING_TOL_DEG:
        turn_mag = int(np.clip(TURN_KP * abs(heading_error_deg), TURN_CMD_MIN, TURN_CMD_MAX))

        # Positive heading error means target is CCW from current heading.
        # This assumes left negative / right positive turns CCW. If it turns the
        # wrong direction on hardware, swap these signs.
        if heading_error_deg > 0:
            motor_left = -turn_mag
            motor_right = turn_mag
            mode = "turn_ccw"
        else:
            motor_left = turn_mag
            motor_right = -turn_mag
            mode = "turn_cw"
    else:
        motor_left = FORWARD_CMD
        motor_right = FORWARD_CMD
        mode = "forward"

    debug_info = {
        "dx": dx,
        "dy": dy,
        "distance_cm": distance_cm,
        "target_heading_deg": target_heading_deg,
        "heading_error_deg": heading_error_deg,
        "mode": mode,
    }
    return int(motor_left), int(motor_right), debug_info


def send_motor_command(ser, motor_left, motor_right):
    """Send ESP32 channel commands ("leftSpeed rightSpeed") over serial."""
    if ser is None:
        return

    try:
        cmd = f"{motor_left} {motor_right}\n"
        ser.write(cmd.encode())
    except Exception as e:
        print(f"[ERROR] Failed to send serial: {e}")


def send_policy_action(ser, action):
    """Scale a normalized policy action to ESP32 motor commands and send it."""
    motor_left, motor_right = esp32_motor_action(action)
    send_motor_command(ser, int(motor_left), int(motor_right))
    return int(motor_left), int(motor_right)


def parse_args():
    """Historical function name; settings are read from JSON, not CLI arguments."""
    from types import SimpleNamespace
    from watercraft.config import load
    c = load('real')
    return SimpleNamespace(controller=c['controller'], weights=c['policy_weights'],
        target=[v * 100 for v in c['goal']], max_action=c['max_action'], port=c['serial_port'],
        dry_run=not c['arm'], cam=c['camera_index'])


def main():
    args = parse_args()
    if args.controller == "policy" and not args.weights:
        sys.exit("--controller policy needs --weights <file>.npz")
    policy = NumpyPolicy(args.weights) if args.controller == "policy" else None
    target = tuple(args.target)
    p_full_scale = max(FORWARD_CMD, TURN_CMD_MAX)
    print(f"Controller: {args.controller}, target {target} cm, max action {args.max_action}, "
          f"{'DRY RUN' if args.dry_run else 'serial ' + args.port}")

    # load K and D from the file
    calib = np.load(CALIB_FILE)
    K, D  = calib["K"], calib["D"]
    D = D.flatten()

    # setup serial communication to ESP-A
    ser = None
    if not args.dry_run:
        try:
            ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
            print(f"Serial connection opened: {args.port} at {BAUD_RATE} baud")
            time.sleep(1)  # Give ESP time to reset
        except serial.SerialException as e:
            sys.exit(f"Failed to open serial port {args.port}: {e} (use --dry-run to test without it)")

    # setup detector
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICT)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    # setup camera
    # DirectShow on Windows opens USB cameras faster and more reliably than the default backend
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.cam, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam} (try another --cam index).")

    try:
        # find homography matrix H of rankd rim, and img_pts of the corners
        Hmat, img_pts = find_homography_mat(cap, detector, K, D)
        R, t, rvec = find_extrinsic_matrix( img_pts, K, D)

        ret, frame = cap.read()
        if not ret:
            print("Its SO OVER")

        vis, id_to_corners = detect_and_draw_markers(frame, detector)
        Extrinsic_water, t_water, Z_water = find_water_extrinsic(id_to_corners, K, D, R, t)

        pose_adapter = PoseObservationAdapter(position_scale_m=POSITION_SCALE_M)
        last_control = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cooked")
                break

            frame = cv2.undistort(frame, K, D)
            vis, id_to_corners = detect_and_draw_markers(frame, detector)
            draw_rectangle_edges(vis, img_pts)

            marker_pose = None
            if Extrinsic_water is not None:
                marker_pose = locate_marker_in_water_frame(
                    id_to_corners, ID_INSIDE, K, D, R, t_water)

            if marker_pose is not None:
                xw, yw, zw, yaw_deg = marker_pose
                cv2.putText(vis, f"ID {ID_INSIDE}: x={xw:.2f} y={yw:.2f} z={zw:.2f} yaw={yaw_deg:.1f}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Control at a fixed rate matching sim training (CONTROL_PERIOD_S). A command is sent
            # every tick even if unchanged: the transmitter stops the motors if serial goes quiet.
            now = time.monotonic()
            if now - last_control >= CONTROL_PERIOD_S:
                last_control = now
                if marker_pose is None:
                    pose_adapter.reset()  # don't difference velocity across the gap
                    action, mode, extra = np.zeros(2), "marker lost -> stop", ""
                else:
                    policy_obs = pose_adapter.update(xw, yw, yaw_deg, now, target[0], target[1])
                    p_left, p_right, debug = compute_motor_command_to_target(xw, yw, yaw_deg, target)
                    if debug["distance_cm"] <= STOP_DISTANCE_CM:
                        action, mode = np.zeros(2), "at goal -> stop"
                    elif policy is not None:
                        action, mode = policy.act(policy_obs), "policy"
                    else:
                        # P-controller output is physical left/right with FORWARD_CMD as full thrust
                        action, mode = np.array([p_left, p_right]) / p_full_scale, "p:" + debug["mode"]
                    extra = (f"pose=({xw:.1f}, {yw:.1f}) cm yaw={yaw_deg:.1f} "
                             f"dist={debug['distance_cm']:.1f}cm err={debug['heading_error_deg']:.1f}deg "
                             f"obs={[round(float(v), 3) for v in policy_obs]}")

                action = np.clip(action, -args.max_action, args.max_action)
                esp_left, esp_right = send_policy_action(ser, action)
                print(f"[{mode}] action=[{action[0]:+.2f} {action[1]:+.2f}] -> ESP L={esp_left} R={esp_right} {extra}")

            cv2.imshow("track_aruco", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q in the OpenCV window
                break
    finally:
        # always leave the motors stopped: normal exit, Ctrl-C or a crash
        if ser is not None:
            for _ in range(3):
                send_motor_command(ser, 0, 0)
                time.sleep(0.05)
            ser.close()
            print("Motors stopped, serial connection closed")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
