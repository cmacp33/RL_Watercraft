import cv2
import os


# use the 4x4 aruco dictionary
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# folder for markers
os.makedirs("markers", exist_ok=True)

ids = [4]
edgePixel = 600

for id in ids:
    marker = aruco.generateImageMarker(dictionary, id, edgePixel)
    
    cv2.imwrite(f"markers/aruco_{id}.png", marker)
    print(f"saved markers/aruco_{id}.png")
    
