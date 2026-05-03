import cv2
import numpy as np
import glob

images_path_glob = "/Users/tracysun/Projects/fiducial_tracking/markers/Calibration_images/calibration_20260327*.jpg"
image_paths = sorted(glob.glob(images_path_glob))
squaresX = 11
squaresY = 8
square_size = 0.02 # in meteres
marker_size = 0.0145 # in meters

DICT = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT) # the aruco dictionary

# generate what the baord will be like
board = cv2.aruco.CharucoBoard((squaresX, squaresY), square_size, marker_size, aruco_dict)
# this was the issue
board.setLegacyPattern(True)

# # yes i am using the correct squaresX and squares Y values, this code below can generate a CharUco board
img = board.generateImage((1100, 800))
cv2.imshow("generated", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

detector_params = cv2.aruco.DetectorParameters()
charuco_params = cv2.aruco.CharucoParameters()

aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

all_charuco_corners = []
all_charuco_ids = []


for p in image_paths:
    
    frame = cv2.imread(p)
    
    if frame is None: 
        print(f"frame is NONE for {p}")
        continue
    
    frame_size = (frame.shape[1], frame.shape[0])
    print(frame_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict) # this is ARUCO markers
    
    if marker_ids is not None and len(marker_ids) > 0:
        print(f"for {p}, marker_id length is{len(marker_ids)}.")
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
        )
        
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        
    else:
        charuco_corners, charuco_ods = None, None
        
# Initialize a guess for the matrix
camera_matrix_init = np.eye(3, dtype=np.float64)
dist_coeffs_init = np.zeros((1, 5), dtype=np.float64)

print (all_charuco_corners)

# reshape to what Charuco expects
all_charuco_corners = [c.astype(np.float32) for c in all_charuco_corners]
all_charuco_ids = [i.astype(np.int32).reshape(-1, 1) for i in all_charuco_ids]
    
flags = 0

proj_error, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=frame_size,
    cameraMatrix=camera_matrix_init,
    distCoeffs=dist_coeffs_init,
    flags=flags,
)

print(f"projection error {proj_error}")
print(f"camera Matrix \n {K}")
print(f"distortion Coeff {D}")

# Verifying Deteced location vs Projected Location

test_image_path = "/Users/tracysun/Projects/fiducial_tracking/markers/Calibration_images/calibration_TEST.jpg"

test_image = cv2.imread(test_image_path)

if test_image is not None:
    
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    frame_show = test_image.copy()
    
else:
    print()

marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(test_gray, aruco_dict)

retval, test_charuco_corners, test_charuco_ids = cv2.aruco.interpolateCornersCharuco(
    marker_corners, marker_ids, test_gray, board
)


all_obj_points = board.getChessboardCorners()

# need to match the N count of test_charuco_ids
obj_points = all_obj_points[test_charuco_ids.flatten()]
matched_img_points = test_charuco_ids

# print(obj_points[0])
# print(test_charuco_corners[0])

# solve for the rvec and tvec
retval, test_rvec, test_tvec = cv2.solvePnP(
    objectPoints = obj_points,
    imagePoints =  test_charuco_corners,
    cameraMatrix=K, 
    distCoeffs=D, 
    rvec = None, 
    tvec = None,
    useExtrinsicGuess=False,
    flags=cv2.SOLVEPNP_ITERATIVE
    )

# list of projected points for all object points
projected_all, _ = cv2.projectPoints(
    obj_points.astype(np.float32),
    test_rvec, test_tvec, K, D
)

# ONLY points that we want to display
projected_id_to_pt = {
    int(test_charuco_ids.flatten()[i]): projected_all[i][0]
    for i in range(len(test_charuco_ids))
}

detected_id_to_pt = {
    int(test_charuco_ids.flatten()[i]): test_charuco_corners[i][0]
    for i in range(len(test_charuco_ids))
}

Verify_IDs = [5, 10, 33, 44, 55]

for corner_id in Verify_IDs:

    if corner_id not in detected_id_to_pt:
        print(f"ID {corner_id} not detected, skipping.")
        continue

    # DETECTED — look up by ID, not by array index
    det_pt = detected_id_to_pt[corner_id]
    x_coord = int(det_pt[0])
    y_coord = int(det_pt[1])

    cv2.circle(frame_show, (x_coord, y_coord), radius=20, color=(0, 255, 0), thickness=-1)

    # PROJECTED
    proj_pt = projected_id_to_pt[corner_id]
    proj_x = int(proj_pt[0])
    proj_y = int(proj_pt[1])

    cv2.circle(frame_show, (proj_x, proj_y), radius=20, color=(0, 0, 255), thickness=-1)

    error = float(np.linalg.norm(det_pt - proj_pt))
    print(f"ID {corner_id} | detected {(x_coord, y_coord)} | projected {(proj_x, proj_y)} | error {error:.2f}px")
    print(error)

print (f"HELLOOOOOOOO{K}")
print (f"DISTORTION COEFFS: {D}")
np.savez("/Users/tracysun/Projects/fiducial_tracking/src/camera_calibration.npz", K=K, D=D)
print("Saved K and D to camera_calibration.npz")

cv2.imshow("DETECTED POINT", frame_show)
cv2.waitKey(0)
cv2.destroyAllWindows()