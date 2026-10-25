import cv2
import numpy as np
from collections import defaultdict
import math
import serial
import time



# ====== SERIAL Settings
SERIAL_PORT    = "/dev/cu.usbserial-140"   #this is for linux/macm change to COM etc on Windows
BAUD_RATE      = 115200
SEND_THRESHOLD = 0.05             # 5% change triggers a send

# ===== INPUT =====
CAM_INDEX = 0
DICT = cv2.aruco.DICT_4X4_50
CALIB_FILE = "src/camera_calibration.npz" # saved from charuco_calibration.py file 

TARGET = (25, 25)

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

_half = MARKER_LENGTH_CM / 2.0
MARKER_OBJ_PTS = np.array([
    [-_half,  _half, 0],   # top-left
    [ _half,  _half, 0],   # top-right
    [ _half, -_half, 0],   # bottom-right
    [-_half, -_half, 0],   # bottom-left
], dtype=np.float64)

RIM_WORLD_PTS_3D = np.array([
    [0.0, 0.0, 0.0],  # BL
    [W,   0.0, 0.0],  # BR
    [W,   H,   0.0],  # TR
    [0.0, H,   0.0],  # TL
], dtype=np.float32)

#====== HELPER FUNCTIONS =======

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

def find_extrinsic_matrix(id_to_corners, K, D):
    
    # initiate img points and world points lists
    img_pts_2d   = []
    world_pts_3d = []
    
    for i, cid in enumerate(CORNER_IDS):
        if cid in id_to_corners:
            centre = marker_center(id_to_corners[cid])   # (x,y) pixel
            img_pts_2d.append(centre) # this is (x,y) in pixel
            world_pts_3d.append(RIM_WORLD_PTS_3D[i]) # this is (x,y,z) in cm
    
    # convert them noth from lists to nump arryys for solvePnP
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
    t    = tvec.flatten()   # (3,1) → (3,)
    
    print(f"[Extrinsic] R:\n{R}")
    print(f"[Extrinsic] t: {t}")
    print(f"[Extrinsic] Full [R|t]:\n{np.hstack([R, t.reshape(3,1)])}")

    return R, t, rvec

def find_water_extrinsic (id_to_corners, K, D, R, t):
    print(f"5 print D matrix {D}, {type(D)},{D.shape}")
    if WATER_ID not in id_to_corners:
        print(f"Marker ID {WATER_ID} not detected, cannot find water line.")
        return None, None, None
    
    img_pts = id_to_corners[WATER_ID].astype(np.float64)   # (4,2)
    
    ok, _, water_tvec = cv2.solvePnP(
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

    print(f"[Water] p_cam:   {p_cam}")
    print(f"[Water] p_world: {p_world}")
    print(f"[Water] Position: x={x_water:.4f}, y={y_water:.4f}, Z={Z_water:.4f} cm")

    return x_water, y_water, Z_water
    
def water_homography(R, t_water, K):
    
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

    # invert H_water to go world → image
    H_inv = np.linalg.inv(H_water)

    # project world corners → image pixels
    img_pts = cv2.perspectiveTransform(
        corners_world.reshape(1, -1, 2), H_inv
    ).reshape(4, 2).astype(int)

    # draw BL→BR→TR→TL→BL
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(vis, tuple(img_pts[a]), tuple(img_pts[b]), (0, 200, 255), 2)

    centre = img_pts.mean(axis=0).astype(int)
    cv2.putText(vis, "water plane", tuple(centre),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def main():
    
    # load K and D from the file
    calib = np.load(CALIB_FILE)
    K, D  = calib["K"], calib["D"]
    D = D.flatten()
    print(f"Loaded K and D from {CALIB_FILE}")
    
    print(f"print K matrix {K}")
    print(f"1 print D matrix {D}, {type(D)},{D.shape}")

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

    # compute H here
    Hmat, img_pts = find_homography_mat(cap, detector, K, D)

    # print(f"2 print D matrix {D}, {type(D)},{D.shape}")

    # tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # undistort every frame before detection
        frame_u = cv2.undistort(frame, K, D)
        vis, id_to_corners = detect_and_draw_markers(frame_u, detector)

        # draw the ROI rectangle using the averaged corner positions
        draw_rectangle_edges(vis, img_pts)
        
        R, t, rvec = find_extrinsic_matrix(id_to_corners, K, D)
        
        # print(f"4 print D matrix {D}, {type(D)},{D.shape}")
        
        H_water = None

        if R is not None:
            
            print(f"just before print D matrix {D}, {type(D)},{D.shape}")
            R_w, t_water, Z_water = find_water_extrinsic(id_to_corners, K, D, R, t)

            if R_w is not None:
                # build new homography at water plane Z
                H_water = water_homography(R_w, t_water, K)
                
                draw_water_plane_rectangle(vis, H_water)
                
                w_rect = map_to_rect(id_to_corners, ID_INSIDE, H_water)
                print(w_rect) 
                print(f"Z_water: {Z_water:.2f} cm")
                cv2.putText(vis, f"Z_water: {Z_water:.2f} cm",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            else:
                cv2.putText(vis, "Water marker (ID5) not found",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(vis, "Rim markers not found — no extrinsic",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # TEMP: Test with regular Hmat instead of H_water
        # H = H_water if H_water is not None else Hmat
        Hom = Hmat  # Force regular homography for testing
        
        # map ID_INSIDE to real-world coords -> called p_rect here (p rectangle LOL)       
        p_rect = map_to_rect(id_to_corners, ID_INSIDE, Hom)

        if p_rect is not None:
            x_cm, y_cm = p_rect
            
            # VALIDATION: Check if values are within reasonable bounds
            # If out of bounds, fall back to regular Hmat
            if x_cm < -10 or x_cm > (W + 10) or y_cm < -10 or y_cm > (H + 10):
                print(f"[WARNING] H_water produced out-of-bounds coords ({x_cm:.2f}, {y_cm:.2f}), falling back to Hmat")
                p_rect = map_to_rect(id_to_corners, ID_INSIDE, Hmat)
                if p_rect is not None:
                    x_cm, y_cm = p_rect
            
            if p_rect is not None:
                x_cm, y_cm = p_rect
            inside = (0 <= x_cm <= W) and (0 <= y_cm <= H)
            
            # force it to be true for testing now
            inside = True
            
            # if it is on the LEFT of target, it is NEGATIVE
            x_diff = x_cm - TARGET[0]
            
            # if it is on TOP of the target, it is POSITIVE
            y_diff = y_cm - TARGET[1]
            
            # print(x_diff, y_diff)
                
            
            # Proportional magnitude that sums up to 1
            total = abs(x_diff) + abs(y_diff)
            x_mag = abs(x_diff) / total
            y_mag = abs(y_diff) / total
            
            if y_diff < 0:
                boat_l, boat_r = abs(x_mag), abs(y_mag) 
            
            if y_diff > 0:
                boat_l, boat_r = -x_mag, -y_mag
                
            distance = math.sqrt(x_diff**2 + y_diff**2)
            
            if distance <= 5:
                boat_l, boat_r = 0,0
                
            print(f"boat_l: {boat_l:.4f}, boat_r: {boat_r:.4f}")
            
            # Convert normalized motor values (-1 to +1) to motor speed range (-255 to +255)
            # Clamp to valid range
            motor_left = int(np.clip(boat_l * 200, -255, 255))
            motor_right = int(np.clip(boat_r * 200, -255, 255))
            
            print(f"Motor commands - Left: {motor_left}, Right: {motor_right}")
            # Send to ESP-A over serial in format: "left right\n"
            if ser is not None:
                try:
                    cmd = f"{motor_left} {motor_right}\n"
                    ser.write(cmd.encode())
                except Exception as e:
                    print(f"[ERROR] Failed to send serial: {e}")
            
            cv2.putText(vis, f"Difference to Target: {x_diff:.4f}, {y_diff:.4f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(vis, f"BOAT L, R {boat_l:.4f}, {boat_r:.4f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (00, 255, 0), 5)
            
            cv2.putText(vis, f"Motor L: {motor_left}, R: {motor_right}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 0), 2)

            print(f"\rID{ID_INSIDE}  x={x_cm:.2f} cm  y={y_cm:.2f} cm  inside={inside}    ", end="")
            

            cv2.putText(vis, f"ID{ID_INSIDE}: x={x_cm:.2f} cm, y={y_cm:.2f} cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"inside ROI: {inside}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print(f"\rWaiting for ID{ID_INSIDE}...    ", end="")
            cv2.putText(vis, f"Waiting for ID{ID_INSIDE}...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # warn if any corner marker has gone missing
        missing = [i for i in CORNER_IDS if i not in id_to_corners]
        if missing:
            cv2.putText(vis, f"Corner missing: {missing}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("track_aruco", vis)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
        print("Serial connection closed")

if __name__ == "__main__":
    main()


