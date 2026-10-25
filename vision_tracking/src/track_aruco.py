import cv2
import os
import numpy as np

'''
This script identifies and marks each corner (x,y) of a defined rectangle
with 4 aruco codes, id 0,1,2,3. Must be placed correctly.

Tracks id 4 aruco and returns position (x,y) with respect to rectangle.

This uses 2D only homography, no 3D pose estimation.
'''
# ===== INPUT =====
CAM_INDEX = 0
DICT = cv2.aruco.DICT_4X4_50

FRAME_W, FRAME_H = 1920, 1080

ID_BL = 0  # bottom-left
ID_BR = 1  # bottom-right
ID_TR = 2  # top-right
ID_TL = 3  # top-left

ID_INSIDE = 4 # object to track

N_CALIB_FRAMES = 10

W = 50 #cm
H = 30 #cm


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

    id_to_corners = {}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_out, corners, ids)

        # Build dict: marker_id -> corners (4,2)
        for c, marker_id in zip(corners, ids.flatten()): # flatten 
            id_to_corners[int(marker_id)] = c[0]  # c is (1,4,2), so now if we have c[0] we only have (4,2) left

        # Draw center dots + labels
        for marker_id, pts in id_to_corners.items():
            center = marker_center(pts).astype(int)
            cv2.circle(frame_out, tuple(center), 5, (0, 255, 0), -1)
            cv2.putText(frame_out, f"ID {marker_id}", (center[0] + 10, center[1]), # c[0] is x, c[1] is y
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    # print("=======HELLO=======")
    # print(id_to_corners)
    return frame_out, id_to_corners


def compute_rect_homography(id_to_corners, W, H,
                            id_bl=ID_BL, id_br=ID_BR, id_tr=ID_TR, id_tl=ID_TL):
    '''
    computes homography for rectangle
    '''
    
    """
    Uses IDs (BL, BR, TR, TL) to compute homography that maps:
        image pixel coords -> rectangle coords (0..W, 0..H)

    Returns:
        Hmat (3x3) or None if not enough markers are present
        img_pts (4x2) in the order [BL, BR, TR, TL] (for drawing), or None
    """
    needed = [id_bl, id_br, id_tr, id_tl]
    if not all(i in id_to_corners for i in needed):
        print("do not have all required IDs")
        return None, None

    # the 4 corner's center point
    img_pts = np.array([
        marker_center(id_to_corners[id_bl]),
        marker_center(id_to_corners[id_br]),
        marker_center(id_to_corners[id_tr]),
        marker_center(id_to_corners[id_tl]),
    ], dtype=np.float32)

    # Rectangle coordinate targets
    rect_pts = np.array([
        [0.0, 0.0],  # BL
        [W,   0.0],  # BR
        [W,   H],    # TR
        [0.0, H],    # TL
    ], dtype=np.float32)

    # find homography
    Hmat, _ = cv2.findHomography(img_pts, rect_pts)
    return Hmat, img_pts

def map_marker_to_rect(id_to_corners, marker_id, Hmat):
    '''
    Maps a marker center from image pixels to rectangle coords using homography.
    Returns its yaw relative to the 
    '''
    
    if Hmat is None or marker_id not in id_to_corners:
        print(f"Cannot map ID{marker_id}, missing homography or marker.")
        print("Hmat:", Hmat)
        return None

    # find center
    p_px = marker_center(id_to_corners[marker_id]).astype(np.float32)  # [x,y]
    
    # find the 4 corners of the marker
    marker_corners = id_to_corners[marker_id]
    corners_trans = cv2.perspectiveTransform(marker_corners.reshape(1, 4, 2), Hmat)[0]  # (4,2)
    # need to reshape p_px to [[[x,y]]], then H transform
    p_rect = cv2.perspectiveTransform(p_px.reshape(1, 1, 2), Hmat)[0, 0] # gets actual point
    TL, TR, BR, BL = corners_trans[0], corners_trans[1], corners_trans[2], corners_trans[3]
    
    v = BR - BL 
    yaw_rad = np.arctan2(v[1], v[0])          # CCW from +x
    yaw_deg = float(np.degrees(yaw_rad))
    # find yaw
    print(f"ID{marker_id} yaw: {yaw_deg:.2f} deg")
    return p_rect, yaw_deg


def draw_rectangle_edges(frame, img_pts_bl_br_tr_tl):
    """Draw rectangle edges on the frame using [BL, BR, TR, TL] order."""
    pts = img_pts_bl_br_tr_tl.astype(int)
    # BL->BR->TR->TL->BL
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 0, 255), 2)


def main():
    
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICT)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    
    # extrinsic calibration once
    read, frame = cap.read()
    if not read:
        print("Cannot open camera")
        return
    
    vis, id_to_corners = detect_and_draw_markers(frame, detector)
    Hmat, img_pts = compute_rect_homography(id_to_corners, W, H)
    Hmat, img_pts = compute_rect_homography(id_to_corners, W, H)
    Hmat, img_pts = compute_rect_homography(id_to_corners, W, H)

    while True:

        read, frame = cap.read()
        vis, id_to_corners = detect_and_draw_markers(frame, detector)
        
        
        if Hmat is not None and img_pts is not None:
            
            print(Hmat)
            
            draw_rectangle_edges(vis, img_pts)

            # map ID4 and constantly print its location
            result = map_marker_to_rect(id_to_corners, ID_INSIDE, Hmat)

            if result is not None:
                p_rect, yaw_deg = result
                # Print continuously (clean: overwrite one line)
                print(f"\rID{ID_INSIDE} rect coords: x={p_rect[0]:2f}, y={p_rect[1]:.2f}", end="")

                # Also display on video
                cv2.putText(vis, f"ID{ID_INSIDE}: x={p_rect[0]:.2f}, y={p_rect[1]:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # inside check
                inside = (0 <= p_rect[0] <= W) and (0 <= p_rect[1] <= H)
                cv2.putText(vis, f"inside: {inside}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f"Waiting for ID{ID_INSIDE}...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            missing = [i for i in [ID_BL, ID_BR, ID_TR, ID_TL] if i not in id_to_corners]
            cv2.putText(vis, f"Missing: {missing}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("track_aruco", vis)
        if (cv2.waitKey(1) & 0xFF) == 27:  # press escape to exit
            break

    print()  # newline after the \r printing
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
    



