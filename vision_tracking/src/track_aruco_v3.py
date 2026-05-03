import cv2
import numpy as np
from collections import defaultdict
import math
import serial
import time

# ====== SERIAL Settings
SERIAL_PORT    = "/dev/cu.usbserial-2140"   #this is for linux/macm change to COM etc on Windows
BAUD_RATE      = 115200
SEND_THRESHOLD = 0.05             # 5% change triggers a send

# ===== INPUT =====
CAM_INDEX = 0
DICT = cv2.aruco.DICT_4X4_50
CALIB_FILE = "src/camera_calibration.npz" # saved from charuco_calibration.py file 

TARGET = (25, 25)
STOP_DISTANCE_CM = 8.0
HEADING_TOL_DEG = 12.0
FORWARD_CMD = 100
TURN_CMD_MIN = 50
TURN_CMD_MAX = 90
TURN_KP = 1.0

# take the average of how many average frames to use for calibration
N_CALIB_FRAMES = 10 

FRAME_W, FRAME_H = 1920, 1080

ID_BL = 0  # bottom-left
ID_BR = 1  # bottom-right
ID_TR = 2  # top-right
ID_TL = 3  # top-left

ID_INSIDE = 4 # object to track
WATER_ID = 4 # id on the water surfacel used this for extrinsic calibration

W = 37.0 # cm
H = 27.0 # cm

CORNER_IDS = [ID_BL, ID_BR, ID_TR, ID_TL]

MARKER_LENGTH_CM = 4.0

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
    """Send left/right motor commands to the controller over serial."""
    print(f"Motor commands - Left: {motor_left}, Right: {motor_right}")

    if ser is None:
        return

    try:
        cmd = f"{motor_left} {motor_right}\n"
        ser.write(cmd.encode())
    except Exception as e:
        print(f"[ERROR] Failed to send serial: {e}")


def main():
    
    # load K and D from the file
    calib = np.load(CALIB_FILE)
    K, D  = calib["K"], calib["D"]
    D = D.flatten()

    # setup serial communication to ESP-A
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial connection opened: {SERIAL_PORT} at {BAUD_RATE} baud")
        time.sleep(1)  # Give ESP time to reset
    except serial.SerialException as e:
        print(f"[WARNING] Failed to open serial port {SERIAL_PORT}: {e}")
        ser = None

    # setup detector
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICT)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    
    # setup camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    
    # find homography matrix H of rankd rim, and img_pts of the corners
    Hmat, img_pts = find_homography_mat(cap, detector, K, D)
    
    
    
    R, t, rvec = find_extrinsic_matrix( img_pts, K, D)
    
    
    ret, frame = cap.read()
    if not ret:
        print("Its SO OVER")
    
    frame_u = cv2.undistort(frame, K, D)
    vis, id_to_corners = detect_and_draw_markers(frame, detector)
    
    Extrinsic_water, t_water, Z_water = find_water_extrinsic(id_to_corners, K, D, R, t)
    
    motor_left, motor_right = 0, 0  # default to stop
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cooked")
            break
        
        frame = cv2.undistort(frame, K, D)
        vis, id_to_corners = detect_and_draw_markers(frame, detector)
        draw_rectangle_edges(vis, img_pts)
        
        
        
        if Extrinsic_water is not None:
            marker_pose = locate_marker_in_water_frame(
                id_to_corners,
                ID_INSIDE,
                K,
                D,
                R,
                t_water,
            )

            if marker_pose is not None:
                xw, yw, zw, yaw_deg = marker_pose
                OLD_motor_left, OLD_motor_right = motor_left, motor_right
                motor_left, motor_right, debug = compute_motor_command_to_target(
                    xw, yw, yaw_deg, TARGET)
                
                motor_left = motor_left * 1.8
                
                # swap left and right if motors reversed
                temp = motor_left
                motor_left = motor_right
                motor_right = temp
                
                print(f"OLD MOTOR CMD: L={OLD_motor_left} R={OLD_motor_right}  →  NEW MOTOR CMD: L={motor_left} R={motor_right}")

                if OLD_motor_left >= 0 and motor_left < 0 or OLD_motor_right >= 0 and motor_right < 0:
                    send_motor_command(ser, motor_left, motor_right)
                    time.sleep(0.1)
                    # send_motor_command(ser, 0, 0)
                    time.sleep(0.1)
                    send_motor_command(ser, motor_left, motor_right)
                
                else:
                    send_motor_command(ser, motor_left, motor_right)
                
                text_overlay = (
                    f"ID {ID_INSIDE}: x={xw:.2f} y={yw:.2f} z={zw:.2f} yaw={yaw_deg:.1f}"
                )
                cv2.putText(
                    vis,
                    text_overlay,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                print(
                    f"({xw:.4f}, {yw:.4f}, {zw:.4f}) cm, {yaw_deg:.2f} deg"
                    f"{debug['mode']} dist={debug['distance_cm']:.1f}cm "
                    f"err={debug['heading_error_deg']:.1f}deg "
                    f"L={motor_left} R={motor_right}"
                )
            else:
                print(f"[Marker {ID_INSIDE}] not detected in current frame"
)

        cv2.imshow("track_aruco", vis)
        

        key = cv2.waitKey(500) & 0xFF
        
        if key == 27:  # ESC via OpenCV window
            # send_motor_command(ser, 0, 0)
            break
    
    cap.release()
    cv2.destroyAllWindows()


    
    if ser is not None:
        cmd = f"{0} {0}\n"
        ser.write(cmd.encode())
        time.sleep(1)
        
        ser.close()
        print("Serial connection closed")


if __name__ == "__main__":
    main()